#!/usr/bin/env python3
"""
kmeansort — Sort photos into folders via k-means clustering on GPS/date EXIF data.

Usage:
    kmeansort --kloc K [--kdate K] [--dry-run|-n] <inputdir> [outputdir]
"""

import argparse
import json
import math
import re
import shutil
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from sklearn.cluster import KMeans


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.tif', '.tiff', '.heic', '.heif', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v'}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
GPS_IFD_TAG = 0x8825


# ─── EXIF ────────────────────────────────────────────────────────────────────


def read_exif(path: Path) -> dict:
    """Return tag-name→value dict from image EXIF; empty dict on any failure."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return {}
        result = {TAGS.get(tid, str(tid)): v for tid, v in exif.items()}
        gps_ifd = exif.get_ifd(GPS_IFD_TAG)
        if gps_ifd:
            result['GPSInfo'] = dict(gps_ifd)
        return result
    except Exception:
        return {}


def _dms_to_decimal(dms, ref: str) -> Optional[float]:
    try:
        d, m, s = (float(x) for x in dms)
        v = d + m / 60 + s / 3600
        return -v if ref in ('S', 'W') else v
    except Exception:
        return None


def parse_gps(exif: dict) -> Optional[tuple[float, float]]:
    gps_raw = exif.get('GPSInfo')
    if not gps_raw:
        return None
    gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
    lat = _dms_to_decimal(gps.get('GPSLatitude', ()), gps.get('GPSLatitudeRef', 'N'))
    lon = _dms_to_decimal(gps.get('GPSLongitude', ()), gps.get('GPSLongitudeRef', 'E'))
    return (lat, lon) if lat is not None and lon is not None else None


def parse_date(exif: dict) -> Optional[date]:
    for field in ('DateTimeOriginal', 'DateTimeDigitized', 'DateTime'):
        raw = exif.get(field)
        if raw:
            try:
                return datetime.strptime(raw, '%Y:%m:%d %H:%M:%S').date()
            except ValueError:
                continue
    return None


try:
    import exiftool as _exiftool
    _EXIFTOOL_AVAILABLE = True
except ImportError:
    _EXIFTOOL_AVAILABLE = False

_exiftool_warned = False


def read_video_metadata(path: Path) -> tuple[Optional[tuple[float, float]], Optional[date]]:
    """Extract GPS and date from a video file via PyExifTool."""
    global _exiftool_warned
    if not _EXIFTOOL_AVAILABLE:
        if not _exiftool_warned:
            print('Warning: pyexiftool not installed; video GPS/date will not be extracted. '
                  'Run: pip install pyexiftool', file=sys.stderr)
            _exiftool_warned = True
        return None, None

    try:
        with _exiftool.ExifToolHelper() as et:
            [data] = et.get_metadata([str(path)], params=['-n'])
    except Exception:
        return None, None

    # Composite:GPS* are signed decimal degrees computed by exiftool from raw atoms.
    lat = data.get('Composite:GPSLatitude') or data.get('EXIF:GPSLatitude')
    lon = data.get('Composite:GPSLongitude') or data.get('EXIF:GPSLongitude')
    gps: Optional[tuple[float, float]] = (float(lat), float(lon)) if lat is not None and lon is not None else None

    dt: Optional[date] = None
    for field in ('EXIF:DateTimeOriginal', 'QuickTime:CreateDate', 'QuickTime:TrackCreateDate', 'QuickTime:MediaCreateDate'):
        raw = data.get(field)
        if raw:
            try:
                dt = datetime.strptime(str(raw), '%Y:%m:%d %H:%M:%S').date()
                break
            except ValueError:
                continue

    return gps, dt


_filecache: dict[str, dict] = {}


def get_file_metadata(path: Path) -> tuple[Optional[tuple[float, float]], Optional[date]]:
    """Return (gps, date) for a file, reading from the metadata cache when available."""
    stat = path.stat()
    key = str(path.resolve())
    entry = _filecache.get(key)

    if entry and entry['mtime'] == stat.st_mtime and entry['size'] == stat.st_size:
        lat, lon = entry.get('lat'), entry.get('lon')
        gps = (lat, lon) if lat is not None and lon is not None else None
        raw_date = entry.get('date')
        return gps, (date.fromisoformat(raw_date) if raw_date else None)

    if path.suffix.lower() in VIDEO_EXTENSIONS:
        gps, dt = read_video_metadata(path)
    else:
        exif = read_exif(path)
        gps, dt = parse_gps(exif), parse_date(exif)

    _filecache[key] = {
        'mtime': stat.st_mtime,
        'size': stat.st_size,
        'lat': gps[0] if gps else None,
        'lon': gps[1] if gps else None,
        'date': dt.isoformat() if dt else None,
    }
    return gps, dt


# ─── Geography ───────────────────────────────────────────────────────────────


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bbox_diagonal_km(lats: list[float], lons: list[float]) -> float:
    """Distance between the two most-distant corners of the bounding box."""
    if len(lats) < 2:
        return 0.0
    lo_lat, hi_lat = min(lats), max(lats)
    lo_lon, hi_lon = min(lons), max(lons)
    return max(
        haversine(lo_lat, lo_lon, hi_lat, hi_lon),
        haversine(lo_lat, hi_lon, hi_lat, lo_lon),
    )


def latlon_to_xyz(lat: float, lon: float) -> tuple[float, float, float]:
    """Unit-sphere Cartesian coordinates — makes k-means wrap-around-safe."""
    φ, λ = math.radians(lat), math.radians(lon)
    return math.cos(φ) * math.cos(λ), math.cos(φ) * math.sin(λ), math.sin(φ)


# ─── Reverse geocoding ───────────────────────────────────────────────────────


_geocache: dict[tuple, str] = {}
_last_request: float = 0.0


def reverse_geocode(lat: float, lon: float) -> str:
    """Nominatim reverse geocode; rate-limited to 1 request/second."""
    global _last_request
    key = (round(lat, 3), round(lon, 3))
    if key in _geocache:
        return _geocache[key]

    wait = 1.0 - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)

    try:
        resp = requests.get(
            'https://nominatim.openstreetmap.org/reverse',
            params={'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 10},
            headers={'User-Agent': 'kmeansort/1.0 (photo-cluster/kmeansort.py)'},
            timeout=10,
        )
        _last_request = time.monotonic()
        if resp.ok:
            name = _format_address(resp.json().get('address', {}))
            _geocache[key] = name
            return name
        print(f'  Warning: geocode HTTP {resp.status_code}', file=sys.stderr)
    except requests.RequestException as exc:
        print(f'  Warning: geocode request failed: {exc}', file=sys.stderr)
        _last_request = time.monotonic()

    fallback = f'{lat:.3f}_{lon:.3f}'
    _geocache[key] = fallback
    return fallback


def _format_address(addr: dict) -> str:
    place = next(
        (addr[f] for f in
         ('city', 'town', 'village', 'hamlet', 'municipality', 'suburb', 'county', 'state')
         if f in addr),
        None,
    )
    country = addr.get('country', '')
    parts = [p for p in (place, country) if p]
    raw = '_'.join(parts) if parts else 'Unknown'
    return _sanitize(raw)


def _sanitize(name: str) -> str:
    """Normalize unicode, then collapse anything non-alphanumeric to underscores."""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^a-zA-Z0-9\-]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')


# ─── Cache persistence ───────────────────────────────────────────────────────

CACHE_FILENAME = '.geocache'
FILECACHE_FILENAME = '.filecache'


def load_geocache(cache_file: Path) -> None:
    if not cache_file.exists():
        return
    try:
        data = json.loads(cache_file.read_text())
        for k, v in data.items():
            lat, lon = k.split(',')
            _geocache[(float(lat), float(lon))] = v
        print(f'Loaded {len(data)} cached geocode result(s) from {cache_file}.')
    except Exception as exc:
        print(f'Warning: could not read geocache ({exc}); starting fresh.', file=sys.stderr)


def save_geocache(cache_file: Path) -> None:
    data = {f'{lat},{lon}': name for (lat, lon), name in _geocache.items()}
    cache_file.write_text(json.dumps(data, indent=2))


def load_filecache(cache_file: Path) -> None:
    if not cache_file.exists():
        return
    try:
        data = json.loads(cache_file.read_text())
        _filecache.update(data)
        print(f'Loaded {len(data)} cached file metadata entry/entries from {cache_file}.')
    except Exception as exc:
        print(f'Warning: could not read file cache ({exc}); starting fresh.', file=sys.stderr)


def save_filecache(cache_file: Path) -> None:
    live = {}
    for key, entry in _filecache.items():
        try:
            st = Path(key).stat()
            if st.st_mtime == entry['mtime'] and st.st_size == entry['size']:
                live[key] = entry
        except OSError:
            pass
    cache_file.write_text(json.dumps(live, indent=2))


# ─── Cleanup ─────────────────────────────────────────────────────────────────


def find_empty_dirs(root: Path, trash: Path) -> list[Path]:
    """Return all dirs that would be empty after recursive cleanup, deepest first."""
    candidates = sorted(
        (d for d in root.rglob('*')
         if d.is_dir() and d != trash and not d.is_relative_to(trash)),
        key=lambda d: len(d.parts),
        reverse=True,
    )
    scheduled: set[Path] = set()
    result: list[Path] = []
    for d in candidates:
        if not d.exists():
            continue
        if all(c in scheduled for c in d.iterdir()):
            scheduled.add(d)
            result.append(d)
    return result


def delete_empty_dirs(empty_dirs: list[Path]) -> None:
    for d in empty_dirs:
        try:
            d.rmdir()
        except OSError:
            pass


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='kmeansort',
        description='Sort photos into folders via k-means clustering on GPS location (and optionally date).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Location folders are named via Nominatim reverse geocoding (1 req/s).\n'
            'When --kdate is used, date sub-folders are nested inside location folders.'
        ),
    )
    p.add_argument('--kloc', type=int, required=True, metavar='K',
                   help='Number of geographic location clusters')
    p.add_argument('--kdate', type=int, metavar='K',
                   help='Number of date clusters (uses DateTimeOriginal, whole dates only)')
    p.add_argument('--guess-loc', action='store_true',
                   help='Assign no-GPS images to a location cluster when their EXIF date is '
                        'bracketed by the date range of exactly one location cluster')
    p.add_argument('--ignore-cache', action='store_true',
                   help='Ignore any existing geocode cache (fresh lookups, but still saves results)')
    p.add_argument('--dry-run', '-n', action='store_true',
                   help='Print a summary of what would be done without moving any files')
    p.add_argument('--no-delete-empty', action='store_true',
                   help='Do not delete empty folders after sorting')
    p.add_argument('inputdir', type=Path,
                   help='Directory containing photos to sort')
    p.add_argument('outputdir', type=Path, nargs='?',
                   help='Destination root directory (default: sort in place within inputdir)')
    return p.parse_args()


# ─── Run log ─────────────────────────────────────────────────────────────────


def _nearest_cluster_info(
    img_date: date,
    kloc: int,
    loc_info: dict,
    date_info: dict,
    loc_date_ranges: dict,
    use_kdate: bool,
) -> dict:
    """Return structured info about the cluster whose date range is nearest to img_date."""
    best_dist: Optional[int] = None
    best_loc: Optional[str] = None
    best_date: Optional[str] = None
    for loc_c in range(kloc):
        if use_kdate and loc_c in date_info:
            for info in date_info[loc_c].values():
                lo, hi = info['lo'], info['hi']
                dist = (lo - img_date).days if img_date < lo else (img_date - hi).days if img_date > hi else 0
                if best_dist is None or dist < best_dist:
                    best_dist, best_loc, best_date = dist, loc_info[loc_c]['folder'], info['folder']
        elif loc_c in loc_date_ranges:
            lo, hi = loc_date_ranges[loc_c]
            dist = (lo - img_date).days if img_date < lo else (img_date - hi).days if img_date > hi else 0
            if best_dist is None or dist < best_dist:
                best_dist, best_loc, best_date = dist, loc_info[loc_c]['folder'], None
    if best_loc is None:
        return {'reason': 'no_clusters'}
    return {'reason': 'nearest', 'loc_folder': best_loc, 'date_folder': best_date, 'dist_days': best_dist}


def _log_file_entry(lines: list[str], name: str, loc_folder: Optional[str],
                    date_folder: Optional[str], note: str = '') -> None:
    note_str = f'  [{note}]' if note else ''
    lines.append(f'{name}{note_str} →' if loc_folder else f'{name}{note_str}')
    if loc_folder:
        lines.append(f'   {loc_folder}/')
        if date_folder:
            lines.append(f'       {date_folder}/')


def write_lastrun_log(
    log_path: Path,
    argv: list[str],
    total_files: int,
    gps_count: int,
    guessed_actual: list[tuple[Path, Path]],
    unguessed: list[dict],
    outdir: Path,
    elapsed: float,
) -> None:
    lines = [
        f'Command: {" ".join(argv)}',
        f'Run date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Elapsed time: {elapsed:.1f}s',
        '',
        f'Total media files scanned: {total_files}',
        f'Files with GPS location: {gps_count}',
        f'Files without GPS: {total_files - gps_count}',
    ]
    if guessed_actual:
        lines += ['', f'Guessed via --guess-loc ({len(guessed_actual)}):']
        for src, dst in guessed_actual:
            try:
                parts = dst.parent.relative_to(outdir).parts
            except ValueError:
                parts = dst.parent.parts
            lines.append('')
            _log_file_entry(lines, src.name,
                            parts[0] if parts else None,
                            parts[1] if len(parts) > 1 else None)
    if unguessed:
        lines += ['', f'Files without location ({len(unguessed)}):']
        for r in unguessed:
            lines.append('')
            cm = r.get('_closest_match', {})
            reason = cm.get('reason') if isinstance(cm, dict) else None
            if reason == 'nearest':
                dist = cm['dist_days']
                note = f'{dist}d away' if dist else 'within range'
                _log_file_entry(lines, r['path'].name, cm['loc_folder'], cm.get('date_folder'), note)
            elif reason == 'ambiguous':
                lines.append(f'{r["path"].name}  [ambiguous]')
                for folder in cm['folders']:
                    lines.append(f'   {folder}/')
            else:
                label = 'no EXIF date' if reason == 'no_date' else 'no dated clusters'
                lines.append(f'{r["path"].name}  [{label}]')
    log_path.write_text('\n'.join(lines) + '\n')


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()

    if not args.inputdir.is_dir():
        sys.exit(f'Error: not a directory: {args.inputdir}')

    in_place = args.outputdir is None
    outdir: Path = args.inputdir if in_place else args.outputdir

    if not args.ignore_cache:
        load_geocache(outdir / CACHE_FILENAME)
        load_filecache(outdir / FILECACHE_FILENAME)

    # ── Scan ──────────────────────────────────────────────────────────────────
    print(f'Scanning {args.inputdir} …')
    all_files = sorted(
        p for p in args.inputdir.rglob('*')
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )
    if not all_files:
        sys.exit('No media files found.')
    print(f'Found {len(all_files)} file(s).')

    # ── Extract EXIF / video metadata ─────────────────────────────────────────
    records: list[dict] = []
    no_gps: list[dict] = []

    for path in all_files:
        gps, dt = get_file_metadata(path)
        if gps is None:
            no_gps.append({'path': path, 'date': dt})
        else:
            records.append({'path': path, 'gps': gps, 'date': dt})

    if no_gps:
        print(f'\n{len(no_gps)} file(s) lack GPS data:')
        for r in no_gps:
            print(f'  {r["path"].name}')

    if not records:
        sys.exit('\nNo files with GPS data to cluster.')

    print(f'\n{len(records)} file(s) have GPS data.')

    # ── Location clustering ───────────────────────────────────────────────────
    kloc = min(args.kloc, len(records))
    if kloc < args.kloc:
        print(f'Note: --kloc reduced to {kloc} (fewer geotagged images than requested).')

    xyz = np.array([latlon_to_xyz(*r['gps']) for r in records])
    kloc = min(kloc, len(np.unique(xyz, axis=0)))
    loc_labels = KMeans(n_clusters=kloc, n_init=10, random_state=42).fit_predict(xyz)
    for i, r in enumerate(records):
        r['loc_label'] = int(loc_labels[i])

    # ── Date clustering (optional, per location cluster) ─────────────────────
    # date_info[loc_label][date_label] = {'folder': ...}
    date_info: dict[int, dict[int, dict]] = {}

    if args.kdate:
        epoch = date(1970, 1, 1)
        no_date_total = 0

        for loc_c in range(kloc):
            loc_recs = [r for r in records if r['loc_label'] == loc_c]
            dated_loc = [r for r in loc_recs if r['date'] is not None]
            no_date_total += len(loc_recs) - len(dated_loc)

            if not dated_loc:
                continue

            kd = min(args.kdate, len(dated_loc))
            if kd < args.kdate:
                print(f'Note: location cluster {loc_c + 1}: --kdate reduced to {kd} '
                      f'({len(dated_loc)} dated image(s)).')

            day_nums = np.array([[(r['date'] - epoch).days] for r in dated_loc], dtype=float)
            kd = min(kd, len(np.unique(day_nums)))
            date_labels = KMeans(n_clusters=kd, n_init=10, random_state=42).fit_predict(day_nums)

            for i, r in enumerate(dated_loc):
                r['date_label'] = int(date_labels[i])

            date_info[loc_c] = {}
            for dc in range(kd):
                recs_dc = [r for r in dated_loc if r['date_label'] == dc]
                dates_dc = sorted(r['date'] for r in recs_dc)
                lo, hi = dates_dc[0], dates_dc[-1]
                folder = lo.isoformat() if lo == hi else f'{lo.isoformat()}_to_{hi.isoformat()}'
                date_info[loc_c][dc] = {'folder': folder, 'lo': lo, 'hi': hi}

        if no_date_total:
            print(f'Note: {no_date_total} image(s) have no date — '
                  'placed in their location folder without a date sub-folder.')

        for r in records:
            r.setdefault('date_label', None)

    # ── Geocode cluster centroids → location folder names ─────────────────────
    print('\nReverse-geocoding location cluster centroids …')
    loc_info: dict[int, dict] = {}
    for c in range(kloc):
        recs_c = [r for r in records if r['loc_label'] == c]
        lats = [r['gps'][0] for r in recs_c]
        lons = [r['gps'][1] for r in recs_c]
        centroid = (sum(lats) / len(lats), sum(lons) / len(lons))
        dist_km = bbox_diagonal_km(lats, lons)
        print(f'  Cluster {c + 1}/{kloc}: {len(recs_c)} photos, '
              f'{dist_km:.0f} km spread … ', end='', flush=True)
        geo = reverse_geocode(*centroid)
        folder = f'{geo}_{dist_km:.0f}km'
        loc_info[c] = {'folder': folder, 'dist_km': dist_km, 'count': len(recs_c)}
        print(folder)

    # ── Guess locations for no-GPS images by date bracketing (optional) ───────
    # Runs after geocoding so that multiple clusters that share a date range but
    # resolve to the same folder are treated as unambiguous.
    loc_date_ranges: dict[int, tuple] = {}
    for loc_c in range(kloc):
        loc_dated = [r for r in records if r['loc_label'] == loc_c and r['date'] is not None]
        if loc_dated:
            dates = [r['date'] for r in loc_dated]
            loc_date_ranges[loc_c] = (min(dates), max(dates))

    guessed_no_gps: list[dict] = []
    remaining_no_gps: list[dict] = no_gps

    if args.guess_loc and no_gps:
        guessed_no_gps = []
        remaining_no_gps = []
        for r in no_gps:
            img_date = r.get('date')
            if img_date is None:
                r['_closest_match'] = 'no EXIF date'
                remaining_no_gps.append(r)
                continue

            # Build candidate matches as (loc_c, date_label_or_None, span_days).
            # When --kdate is active, match against individual date sub-cluster
            # windows so span comparisons reflect actual trip windows rather than
            # the full cluster lifetime. Fall back to the overall cluster range
            # only when date sub-clusters are unavailable.
            matches: list[tuple[int, Optional[int], int]] = []
            for loc_c in range(kloc):
                if args.kdate and loc_c in date_info:
                    for dc, info in date_info[loc_c].items():
                        if info['lo'] <= img_date <= info['hi']:
                            matches.append((loc_c, dc, (info['hi'] - info['lo']).days))
                elif loc_c in loc_date_ranges:
                    lo, hi = loc_date_ranges[loc_c]
                    if lo <= img_date <= hi:
                        matches.append((loc_c, None, (hi - lo).days))

            if not matches:
                r['_closest_match'] = _nearest_cluster_info(
                    img_date, kloc, loc_info, date_info, loc_date_ranges, bool(args.kdate)
                )
                remaining_no_gps.append(r)
                continue

            # Prefer the tightest window; a 5-day trip sub-cluster beats a
            # 9-month home cluster that happens to overlap.
            min_span = min(m[2] for m in matches)
            best = [m for m in matches if m[2] == min_span]
            best_folders = {loc_info[m[0]]['folder'] for m in best}
            if len(best_folders) != 1:
                r['_closest_match'] = 'ambiguous: ' + ' or '.join(sorted(best_folders))
                remaining_no_gps.append(r)
                continue

            r['loc_label'] = best[0][0]
            r['date_label'] = best[0][1]
            guessed_no_gps.append(r)

        if guessed_no_gps:
            print(f'\n--guess-loc: {len(guessed_no_gps)} no-GPS image(s) assigned to location clusters by date:')
            for r in guessed_no_gps:
                print(f'  {r["path"].name} → {loc_info[r["loc_label"]]["folder"]}')
        if remaining_no_gps:
            print(f'--guess-loc: {len(remaining_no_gps)} no-GPS image(s) remain unmatched → Unknown_Location/.')

    # ── Annotate remaining no-GPS files with closest cluster (for the log) ─────
    for r in remaining_no_gps:
        if '_closest_match' not in r:
            img_date = r.get('date')
            if img_date is None:
                r['_closest_match'] = 'no EXIF date'
            else:
                r['_closest_match'] = _nearest_cluster_info(
                    img_date, kloc, loc_info, date_info, loc_date_ranges, bool(args.kdate)
                )

    # ── Date clustering for remaining no-GPS items ────────────────────────────
    no_gps_date_info: dict[int, dict] = {}

    if args.kdate and remaining_no_gps:
        epoch = date(1970, 1, 1)
        dated_no_gps = [r for r in remaining_no_gps if r['date'] is not None]

        if dated_no_gps:
            kd = min(args.kdate, len(dated_no_gps))
            if kd < args.kdate:
                print(f'Note: no-location items: --kdate reduced to {kd} '
                      f'({len(dated_no_gps)} dated image(s)).')
            day_nums = np.array([[(r['date'] - epoch).days] for r in dated_no_gps], dtype=float)
            kd = min(kd, len(np.unique(day_nums)))
            date_labels = KMeans(n_clusters=kd, n_init=10, random_state=42).fit_predict(day_nums)

            for i, r in enumerate(dated_no_gps):
                r['date_label'] = int(date_labels[i])

            for dc in range(kd):
                recs_dc = [r for r in dated_no_gps if r['date_label'] == dc]
                dates_dc = sorted(r['date'] for r in recs_dc)
                lo, hi = dates_dc[0], dates_dc[-1]
                folder = lo.isoformat() if lo == hi else f'{lo.isoformat()}_to_{hi.isoformat()}'
                no_gps_date_info[dc] = {'folder': folder}

    for r in remaining_no_gps:
        r.setdefault('date_label', None)

    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
        save_geocache(outdir / CACHE_FILENAME)
        save_filecache(outdir / FILECACHE_FILENAME)

    # ── Plan moves ────────────────────────────────────────────────────────────
    moves: list[tuple[Path, Path]] = []  # (src, dest)

    for r in records + guessed_no_gps:
        loc_folder = loc_info[r['loc_label']]['folder']
        date_label = r.get('date_label')

        if args.kdate and date_label is not None and date_label in date_info.get(r['loc_label'], {}):
            dest_dir = outdir / loc_folder / date_info[r['loc_label']][date_label]['folder']
        else:
            dest_dir = outdir / loc_folder

        moves.append((r['path'], dest_dir / r['path'].name))

    no_loc_dir = outdir / 'Unknown_Location'
    for r in remaining_no_gps:
        date_label = r.get('date_label')
        if date_label is not None and date_label in no_gps_date_info:
            dest_dir = no_loc_dir / no_gps_date_info[date_label]['folder']
        else:
            dest_dir = no_loc_dir
        moves.append((r['path'], dest_dir / r['path'].name))

    # ── Dry-run summary ───────────────────────────────────────────────────────
    trash = outdir / '_trash'

    if args.dry_run:
        print('\n─── DRY RUN ──────────────────────────────────────────────────────')
        by_dir: dict[Path, list[str]] = defaultdict(list)
        for src, dst in moves:
            by_dir[dst.parent].append(src.name)

        total_dirs = len(by_dir)
        for folder in sorted(by_dir):
            files = sorted(by_dir[folder])
            try:
                rel = folder.relative_to(outdir)
            except ValueError:
                rel = folder
            print(f'\n  {rel}/  ({len(files)} file(s))')
            for f in files:
                print(f'    {f}')

        empty = find_empty_dirs(outdir, trash) if (outdir.exists() and not args.no_delete_empty) else []
        if empty:
            print(f'\n  [would delete]  ({len(empty)} empty folder(s))')
            for d in empty:
                print(f'    {d.relative_to(outdir)}/')

        print(f'\nSummary: {len(moves)} file(s) would move into {total_dirs} folder(s) under {outdir}.')
        if guessed_no_gps:
            print(f'         {len(guessed_no_gps)} no-GPS file(s) would be assigned to location clusters (--guess-loc).')
        if remaining_no_gps:
            print(f'         {len(remaining_no_gps)} file(s) (no GPS) would go to Unknown_Location/.')
        if empty:
            print(f'         {len(empty)} empty folder(s) would be deleted.')
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    print(f'\nMoving {len(moves)} file(s) …')
    dirs_created: set[Path] = set()
    actual_moves: list[tuple[Path, Path]] = []
    for i, (src, dst) in enumerate(moves, 1):
        if dst.parent not in dirs_created:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dirs_created.add(dst.parent)

        # Resolve filename collisions
        final = dst
        if final.exists() and final.resolve() != src.resolve():
            j = 1
            while final.exists():
                final = dst.parent / f'{dst.stem}_{j}{dst.suffix}'
                j += 1

        shutil.move(str(src), str(final))
        actual_moves.append((src, final))
        try:
            rel = final.relative_to(outdir)
        except ValueError:
            rel = final
        print(f'  [{i:>{len(str(len(moves)))}}/{len(moves)}] {src.name} → {rel}')

    print(f'\nDone. {len(moves)} file(s) moved into {outdir}.')
    if guessed_no_gps:
        print(f'{len(guessed_no_gps)} no-GPS file(s) were assigned to location clusters (--guess-loc).')
    if remaining_no_gps:
        print(f'{len(remaining_no_gps)} file(s) (no GPS) were placed in Unknown_Location/.')

    # ── Cleanup empty dirs ────────────────────────────────────────────────────
    if not args.no_delete_empty:
        empty = find_empty_dirs(outdir, trash)
        if empty:
            print(f'\nDeleting {len(empty)} empty folder(s) …')
            delete_empty_dirs(empty)
            for d in empty:
                print(f'  {d.relative_to(outdir)}/')

    # ── Write run log ─────────────────────────────────────────────────────────
    guessed_paths = {r['path'] for r in guessed_no_gps}
    guessed_actual = [(src, dst) for src, dst in actual_moves if src in guessed_paths]
    write_lastrun_log(
        outdir / '_lastrun.log',
        sys.argv,
        len(all_files),
        len(records),
        guessed_actual,
        remaining_no_gps,
        outdir,
        time.monotonic() - t0,
    )
    print(f'Run log written to {outdir / "_lastrun.log"}.')


if __name__ == '__main__':
    main()

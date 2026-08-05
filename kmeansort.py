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


# ─── Geocache persistence ────────────────────────────────────────────────────

CACHE_FILENAME = '.geocache'


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


# ─── Cleanup ─────────────────────────────────────────────────────────────────


def find_empty_dirs(root: Path, trash: Path) -> list[Path]:
    """Return empty directories under root (deepest first), excluding trash."""
    candidates = sorted(
        (d for d in root.rglob('*')
         if d.is_dir() and d != trash and not d.is_relative_to(trash)),
        key=lambda d: len(d.parts),
        reverse=True,
    )
    # Check live state so a parent that becomes empty after its children are
    # processed in the same pass is also caught.
    return [d for d in candidates if d.exists() and not any(d.iterdir())]


def move_to_trash(empty_dirs: list[Path], root: Path, trash: Path) -> None:
    for d in empty_dirs:
        dest = trash / d.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dest))


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
    p.add_argument('--ignore-cache', action='store_true',
                   help='Ignore any existing geocode cache (fresh lookups, but still saves results)')
    p.add_argument('--dry-run', '-n', action='store_true',
                   help='Print a summary of what would be done without moving any files')
    p.add_argument('inputdir', type=Path,
                   help='Directory containing photos to sort')
    p.add_argument('outputdir', type=Path, nargs='?',
                   help='Destination root directory (default: sort in place within inputdir)')
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if not args.inputdir.is_dir():
        sys.exit(f'Error: not a directory: {args.inputdir}')

    in_place = args.outputdir is None
    outdir: Path = args.inputdir if in_place else args.outputdir

    if not args.ignore_cache:
        load_geocache(outdir / CACHE_FILENAME)

    # ── Scan ──────────────────────────────────────────────────────────────────
    print(f'Scanning {args.inputdir} …')
    all_files = sorted(
        p for p in args.inputdir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_files:
        sys.exit('No image files found.')
    print(f'Found {len(all_files)} image(s).')

    # ── Extract EXIF ──────────────────────────────────────────────────────────
    records: list[dict] = []
    no_gps: list[Path] = []

    for path in all_files:
        exif = read_exif(path)
        gps = parse_gps(exif)
        if gps is None:
            no_gps.append(path)
        else:
            records.append({'path': path, 'gps': gps, 'date': parse_date(exif)})

    if no_gps:
        print(f'\n{len(no_gps)} image(s) lack GPS data (will be skipped):')
        for p in no_gps:
            print(f'  {p.name}')

    if not records:
        sys.exit('\nNo images with GPS data to cluster.')

    print(f'\n{len(records)} image(s) have GPS data.')

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
                date_info[loc_c][dc] = {'folder': folder}

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

    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
        save_geocache(outdir / CACHE_FILENAME)

    # ── Plan moves ────────────────────────────────────────────────────────────
    moves: list[tuple[Path, Path]] = []  # (src, dest)

    for r in records:
        loc_folder = loc_info[r['loc_label']]['folder']
        date_label = r.get('date_label')

        if args.kdate and date_label is not None and date_label in date_info.get(r['loc_label'], {}):
            dest_dir = outdir / loc_folder / date_info[r['loc_label']][date_label]['folder']
        else:
            dest_dir = outdir / loc_folder

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

        if no_gps:
            print(f'\n  [skipped — no GPS]  ({len(no_gps)} file(s))')
            for p in no_gps:
                print(f'    {p.name}')

        empty = find_empty_dirs(outdir, trash) if outdir.exists() else []
        if empty:
            print(f'\n  [would move to _trash/]  ({len(empty)} empty folder(s))')
            for d in empty:
                print(f'    {d.relative_to(outdir)}/')

        print(f'\nSummary: {len(moves)} file(s) would move into {total_dirs} folder(s) under {outdir}.')
        if no_gps:
            print(f'         {len(no_gps)} file(s) would be left in place (no GPS).')
        if empty:
            print(f'         {len(empty)} empty folder(s) would be moved to _trash/.')
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    print(f'\nMoving {len(moves)} file(s) …')
    dirs_created: set[Path] = set()
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
        try:
            rel = final.relative_to(outdir)
        except ValueError:
            rel = final
        print(f'  [{i:>{len(str(len(moves)))}}/{len(moves)}] {src.name} → {rel}')

    print(f'\nDone. {len(moves)} file(s) moved into {outdir}.')
    if no_gps:
        print(f'{len(no_gps)} file(s) were skipped (no GPS data).')

    # ── Cleanup empty dirs ────────────────────────────────────────────────────
    empty = find_empty_dirs(outdir, trash)
    if empty:
        print(f'\nMoving {len(empty)} empty folder(s) to _trash/ …')
        move_to_trash(empty, outdir, trash)
        for d in empty:
            print(f'  {d.relative_to(outdir)}/')


if __name__ == '__main__':
    main()

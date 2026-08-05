# photo-geocluster

**photo-geocluster** groups a directory of photos and videos into named subfolders using k-means clustering on GPS coordinates and capture dates extracted from EXIF metadata. Each geographic cluster is automatically labelled via Nominatim reverse geocoding, and an optional second clustering pass splits each location into date-based sub-folders — giving you a clean hierarchy like `Portland, Oregon / 2023-07` without any manual tagging.

The author has tested this script but there is NO WARRANTY. It can delete files! Use at your own risk.

## Synopsis

```
kmeansort --kloc K [--kdate K] [--guess-loc] [--dry-run] <inputdir> [outputdir]
```

| Option | Description |
|---|---|
| `--kloc K` | Number of geographic clusters (required) |
| `--kdate K` | Number of date sub-geoclusters per location |
| `--guess-loc` | Assign GPS-less files to a location when their date falls entirely within one cluster's range |
| `--ignore-cache` | Skip the geocode cache and re-fetch all place names |
| `--dry-run`, `-n` | Preview changes without moving any files |
| `--no-delete-empty` | Keep empty folders after sorting |

`outputdir` defaults to sorting in-place inside `inputdir`.

## Quick start

**1. Install `uv` (via Homebrew)**

```bash
brew install uv
```

**2. Clone and run**

```bash
git clone https://github.com/csghormley/photo-geocluster
cd photo-geocluster
uv run kmeansort --kloc 5 --kdate 3 ~/Pictures/Unsorted ~/Pictures/Sorted
```

`uv run` creates an isolated virtual environment and installs dependencies automatically on first use — no manual `pip install` needed.

**Dry-run first** to preview what will move:

```bash
uv run kmeansort --kloc 5 --dry-run ~/Pictures/Unsorted
```

> **Note:** Nominatim reverse geocoding is rate-limited to 1 request per second. Results are cached locally so subsequent runs over the same photos are fast.

## Acknowledgements

- [Claude](https://www.anthropic.com/claude) by Anthropic — this project was developed with the assistance of Claude Code.
- [ExifTool](https://exiftool.org/) by Phil Harvey — the indispensable Swiss Army knife for reading and writing image metadata.
- [OpenStreetMap](https://www.openstreetmap.org/) and the [Nominatim](https://nominatim.org/) geocoding service — for providing free, open reverse geocoding that powers location labelling.

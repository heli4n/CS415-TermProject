# FMA-small Dataset Audit Report

## Dataset Summary
- Original tracks: 8000
- Duplicate track IDs: 0
- Unique artists: 2309
- Artists appearing in multiple genres: 84
- Removed invalid audio tracks: 6
- Remaining tracks: 7994
- Audio validation: Applied bad_files.csv

## Genre Distribution (original 8,000 tracks)

| Genre | Track count |
|---|---:|
| Electronic | 1000 |
| Experimental | 1000 |
| Folk | 1000 |
| Hip-Hop | 1000 |
| Instrumental | 1000 |
| International | 1000 |
| Pop | 1000 |
| Rock | 1000 |

## After removing bad audio (clean_tracks.csv, used for all experiments)
- Tracks: 7994
- Unique artists: 2307
- Unique albums: 2462
- Artists appearing in multiple genres: 84
- Tracks per artist: mean 3.47, median 1.00, max 208

| Genre | Track count |
|---|---:|
| Electronic | 999 |
| Experimental | 999 |
| Folk | 1000 |
| Hip-Hop | 997 |
| Instrumental | 1000 |
| International | 1000 |
| Pop | 1000 |
| Rock | 999 |

## Tracks per Artist (original 8,000 tracks)
- Mean: 3.46
- Median: 1.00
- Maximum: 208

![Artist histogram](artist_histogram.png)

## Notes
- Official FMA-small split labels were preserved.
- Audio exclusions are applied only when bad_files.csv exists.
- The final clean_tracks.csv schema requires team confirmation.

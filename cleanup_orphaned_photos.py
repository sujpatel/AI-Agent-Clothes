"""Maintenance script: deletes photo files in data/photos/ that aren't
referenced by any item in Pinecone and are old enough to safely assume
abandoned (not something mid-review right now).

This covers batches where the user detected items but never confirmed or
discarded them — those crop files would otherwise sit on disk forever.

Usage:
    python cleanup_orphaned_photos.py            # deletes anything orphaned and older than 1 hour
    python cleanup_orphaned_photos.py --dry-run   # just print what would be deleted
    python cleanup_orphaned_photos.py --min-age-hours 6
"""

import argparse
import time
from pathlib import Path

from app.vectorstore import list_items

PHOTOS_DIR = Path(__file__).parent / "data" / "photos"


def find_orphaned_photos(min_age_hours: float) -> list[Path]:
    referenced_filenames = {Path(item["photo_path"]).name for item in list_items() if item.get("photo_path")}

    cutoff = time.time() - (min_age_hours * 3600)
    orphaned = []
    for photo_path in PHOTOS_DIR.iterdir():
        if not photo_path.is_file():
            continue
        if photo_path.name in referenced_filenames:
            continue
        if photo_path.stat().st_mtime > cutoff:
            continue  # too new — might still be mid-review, leave it alone
        orphaned.append(photo_path)

    return orphaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be deleted")
    parser.add_argument("--min-age-hours", type=float, default=1.0)
    args = parser.parse_args()

    orphans = find_orphaned_photos(args.min_age_hours)

    if not orphans:
        print("No orphaned photos found.")
    else:
        print(f"Found {len(orphans)} orphaned photo(s):")
        for path in orphans:
            print(f"  {path.name}")
            if not args.dry_run:
                path.unlink(missing_ok=True)

        if args.dry_run:
            print("\n(dry run — nothing deleted)")
        else:
            print("\nDeleted.")

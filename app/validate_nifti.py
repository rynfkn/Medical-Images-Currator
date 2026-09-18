"""Read-only check of all NIfTI image/mask pairs before ingestion."""

import argparse
import json
from pathlib import Path

from fastapi import HTTPException

from app.services.annotation import nifti_metadata
from app.services.ingestion import nifti_images
from app.services.storage import import_path, source_file


def validate_dataset(path: str) -> dict:
    root = import_path(path)
    images = nifti_images(root)
    errors = []
    seen_uids = set()
    for entry in images:
        try:
            suffix = ".nii.gz" if entry.name.endswith(".nii.gz") else ".nii"
            uid = entry.name.removesuffix(suffix)
            if not uid or len(uid) > 255:
                raise ValueError("Case identifier must contain 1–255 characters")
            if uid in seen_uids:
                raise ValueError(f"Duplicate case identifier: {uid}")
            seen_uids.add(uid)
            image = source_file(root, f"images/{entry.name}")
            mask = source_file(root, f"labels/{entry.name}")
            nifti_metadata(image, mask)
        except (ValueError, OSError) as exc:
            errors.append({"file": entry.name, "error": str(exc)})
    return {
        "path": str(root),
        "cases_checked": len(images),
        "valid_cases": len(images) - len(errors),
        "invalid_cases": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Dataset directory under IMPORT_DIR")
    args = parser.parse_args()
    try:
        report = validate_dataset(str(args.path))
    except (HTTPException, ValueError, OSError) as exc:
        message = exc.detail if isinstance(exc, HTTPException) else str(exc)
        print(json.dumps({"error": message}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 1 if report["invalid_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

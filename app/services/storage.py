import io
import json
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings


def get_file_path(relative_path: str) -> Path:
    root = get_settings().data_dir.resolve()
    path = (root / relative_path).resolve()
    if Path(relative_path).is_absolute() or not path.is_relative_to(root) or path == root:
        raise HTTPException(400, "Invalid dataset file path")
    return path


def save_file(
    db: Session, source: BinaryIO, relative_path: str, *, max_bytes: int | None = None
) -> str:
    path = get_file_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target = path.open("xb")  # Exclusive creation, including original annotations.
    except FileExistsError:
        raise HTTPException(409, "Dataset file already exists")
    db.info.setdefault("created_files", []).append(path)
    with target:
        count = 0
        while chunk := source.read(1024 * 1024):
            count += len(chunk)
            if max_bytes is not None and count > max_bytes:
                raise HTTPException(413, "Annotation exceeds upload size limit")
            target.write(chunk)
    return relative_path


def copy_file(db: Session, source: Path, relative_path: str) -> str:
    with source.open("rb") as stream:
        return save_file(db, stream, relative_path)


def save_json(db: Session, value: dict, relative_path: str) -> str:
    return save_file(db, io.BytesIO(json.dumps(value, allow_nan=False).encode()), relative_path)


def delete_file(relative_path: str) -> None:
    """Internal cleanup utility; never exposed as an annotation deletion API."""
    get_file_path(relative_path).unlink(missing_ok=True)


def import_path(value: str) -> Path:
    root = get_settings().import_dir.resolve()
    path = Path(value).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise HTTPException(400, "Import must be a directory inside IMPORT_DIR")
    if path == get_settings().data_dir.resolve() or path.is_relative_to(
        get_settings().data_dir.resolve()
    ):
        raise HTTPException(400, "Import and managed dataset storage must be separate")
    return path


def source_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if (
        Path(name).is_absolute()
        or not path.is_relative_to(root.resolve())
        or not path.is_relative_to(get_settings().import_dir.resolve())
        or not path.is_file()
    ):
        raise ValueError(f"Missing or unsafe input file: {name}")
    return path

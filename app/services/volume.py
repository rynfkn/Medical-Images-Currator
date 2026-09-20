"""Slice server for the image viewer.

Images and masks are decoded once into uncompressed, RAS-oriented cache files
under ``DATA_DIR`` and read back through numpy memmaps, so a slice request never
has to hold a whole CT volume in memory. Everything the viewer works with is a
3D array indexed ``[x, y, z]`` in RAS order (x to Right, y to Anterior, z to
Superior); 2D raster cases are volumes with a single slice.

ponytail: caches are written next to the case and never pruned. Add an LRU
sweep over ``viewer/`` directories if dataset storage becomes a constraint.
"""

import gzip
import io
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import nibabel as nib
import numpy as np
import pydicom
from fastapi import HTTPException
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation, ornt_transform
from PIL import Image, UnidentifiedImageError
from pycocotools import mask as coco_mask

from app.models import AnnotationFormat, AnnotationVersion, Case, Dataset, ImageFormat
from app.services.annotation import read_json, rle_counts
from app.services.storage import get_file_path

# Orientation of an array that is already in RAS order.
RAS_ORNT = np.array([[0, 1], [1, 1], [2, 1]], dtype=float)
# Read roughly 8M voxels at a time so decoding never allocates a whole volume.
CHUNK_VOXELS = 1 << 23
# Cache files are Fortran-ordered so an axial slice is one contiguous block.
LAYOUT = "F"
AXIS_NAMES = ("Sagittal", "Coronal", "Axial")


@contextmanager
def mappable(path: Path):
    """Yield a path nibabel can memory-map, expanding a gzipped NIfTI to a temp file."""
    if path.name.endswith(".gz"):
        handle = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
        try:
            with gzip.open(path, "rb") as source:
                shutil.copyfileobj(source, handle, 1 << 20)
            handle.close()
            yield Path(handle.name)
        finally:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)
    else:
        yield path


def to_ras(array: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ornt = io_orientation(affine)
    return apply_orientation(array, ornt), ornt, affine @ inv_ornt_aff(ornt, array.shape)


def from_ras(array: np.ndarray, ornt) -> np.ndarray:
    """Return a RAS array in the source file's own axis order."""
    return apply_orientation(np.asarray(array), ornt_transform(RAS_ORNT, np.asarray(ornt, float)))


def read_nifti(path: Path, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a NIfTI file into (RAS array, source affine, source orientation)."""
    limits = np.iinfo(dtype)
    with mappable(path) as usable:
        try:
            image = nib.load(usable)
            if len(image.shape) != 3:
                raise ValueError("Expected a 3D NIfTI volume")
            plane = max(1, int(np.prod(image.shape[1:])))
            step = max(1, CHUNK_VOXELS // plane)
            stacked = np.empty(image.shape, dtype=dtype)
            for start in range(0, image.shape[0], step):
                stop = min(image.shape[0], start + step)
                block = np.asanyarray(image.dataobj[start:stop], dtype=np.float64)
                stacked[start:stop] = np.clip(np.rint(block), limits.min, limits.max)
            affine = np.asarray(image.affine, dtype=float)
        except (OSError, EOFError, nib.filebasedimages.ImageFileError) as exc:
            raise ValueError("Unreadable NIfTI file") from exc
    ras, ornt, _ = to_ras(stacked, affine)
    return ras, affine, ornt


def dicom_affine(first, last, count: int) -> np.ndarray:
    """Build an RAS affine from series geometry, falling back to axis-aligned."""
    spacing = [float(x) for x in getattr(first, "PixelSpacing", (1.0, 1.0))]
    thickness = float(getattr(first, "SliceThickness", 0) or 1.0)
    orientation = [float(x) for x in getattr(first, "ImageOrientationPatient", ())]
    if len(orientation) != 6:
        orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    column, row = np.array(orientation[:3]), np.array(orientation[3:])
    origin = np.array([float(x) for x in getattr(first, "ImagePositionPatient", (0.0, 0.0, 0.0))])
    ending = [float(x) for x in getattr(last, "ImagePositionPatient", ())]
    step = (np.array(ending) - origin) / (count - 1) if count > 1 and len(ending) == 3 else None
    if step is None or not np.isfinite(step).all() or not np.linalg.norm(step):
        step = np.cross(column, row) * thickness
    affine = np.eye(4)
    affine[:3, 0] = column * spacing[1]
    affine[:3, 1] = row * spacing[0]
    affine[:3, 2] = step
    affine[:3, 3] = origin
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine  # DICOM patient space is LPS.


def read_dicom_series(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = [get_file_path(item["path"]) for item in case.metadata_json["slices"]]
    try:
        first = pydicom.dcmread(paths[0])
        volume = np.empty((first.Columns, first.Rows, len(paths)), dtype=np.int16)
        for index, path in enumerate(paths):
            frame = first if index == 0 else pydicom.dcmread(path)
            pixels = frame.pixel_array.astype(np.float32)
            slope = float(getattr(frame, "RescaleSlope", 1) or 1)
            intercept = float(getattr(frame, "RescaleIntercept", 0) or 0)
            volume[:, :, index] = np.clip(np.rint(pixels * slope + intercept), -32768, 32767).T
        last = pydicom.dcmread(paths[-1], stop_before_pixels=True)
        affine = dicom_affine(first, last, len(paths))
    except Exception as exc:  # pydicom surfaces many decoding errors by transfer syntax.
        raise ValueError(f"Unreadable DICOM pixel data: {exc}") from exc
    ras, ornt, _ = to_ras(volume, affine)
    return ras, affine, ornt


def read_raster(path: Path) -> np.ndarray:
    """Load a PNG/JPEG as a single-slice volume whose axial view is the upright image."""
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Unreadable image file") from exc
    return pixels[::-1, ::-1].T[..., None].astype(np.int16)


def image_source(case: Case, dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset.image_format == ImageFormat.DICOM:
        return read_dicom_series(case)
    if dataset.image_format == ImageFormat.NIFTI:
        return read_nifti(get_file_path(case.image_path), np.int16)
    return read_raster(get_file_path(case.image_path)), np.eye(4), RAS_ORNT


@lru_cache(maxsize=6)
def _mapped(path: str, fingerprint: tuple, shape: tuple, dtype: str) -> np.memmap:
    """Re-use one read-only mapping per cache file; `fingerprint` invalidates rewrites."""
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, order=LAYOUT)


def mapped(relative_path: str, shape, dtype) -> np.memmap:
    path = get_file_path(relative_path)
    status = path.stat()
    name = np.dtype(dtype).name
    return _mapped(str(path), (status.st_mtime_ns, status.st_size), tuple(shape), name)


@lru_cache(maxsize=8)
def _meta(path: str, fingerprint: tuple) -> dict:
    """Every slice request needs this; re-reading the JSON each time is wasted I/O."""
    return json.loads(Path(path).read_text())


def store(relative_path: str, array: np.ndarray, dtype) -> np.memmap:
    """Write an array to an uncompressed cache file and map it back read-only."""
    path = get_file_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        buffer = np.memmap(
            temporary, dtype=dtype, mode="w+", shape=tuple(array.shape), order=LAYOUT
        )
        buffer[:] = array
        buffer.flush()
        del buffer
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return mapped(relative_path, array.shape, dtype)


def viewer_dir(case: Case) -> str:
    return f"{case.dataset_id}/cases/{case.id}/viewer"


def image_volume(case: Case, dataset: Dataset) -> tuple[np.memmap, dict]:
    """Return the cached RAS image volume and its viewer metadata, building it once."""
    meta_path = get_file_path(f"{viewer_dir(case)}/image.json")
    data_path = get_file_path(f"{viewer_dir(case)}/image.i16")
    if meta_path.is_file() and data_path.is_file():
        status = meta_path.stat()
        meta = _meta(str(meta_path), (status.st_mtime_ns, status.st_size))
        if meta.get("layout") == LAYOUT:
            return mapped(f"{viewer_dir(case)}/image.i16", meta["shape"], np.int16), meta
    try:
        ras, affine, ornt = image_source(case, dataset)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    ras_affine = affine @ inv_ornt_aff(ornt, from_ras(ras, ornt).shape)
    sample = ras[::4, ::4, ::4].astype(np.float32)
    low, high = (float(value) for value in np.percentile(sample, (0.5, 99.5)))
    if high - low < 1:
        low, high = float(ras.min()), max(float(ras.max()), float(ras.min()) + 1)
    if low < -500:
        # Hounsfield units: the air background would otherwise wash out the window.
        low, high = -160.0, 240.0
    meta = {
        "layout": LAYOUT,
        "shape": [int(x) for x in ras.shape],
        "spacing": [float(x) for x in np.linalg.norm(ras_affine[:3, :3], axis=0)],
        "affine": np.asarray(affine, dtype=float).tolist(),
        "ornt": np.asarray(ornt, dtype=float).tolist(),
        "level": round((high + low) / 2, 2),
        "width": round(max(high - low, 1.0), 2),
        "range": [int(ras.min()), int(ras.max())],
    }
    volume = store(f"{viewer_dir(case)}/image.i16", ras, np.int16)
    meta_path.write_text(json.dumps(meta))
    return volume, meta


def coco_categories(categories: list[dict]) -> dict[int, dict]:
    """Map viewer labels to original categories; zero is reserved for background."""
    if len(categories) > 255:
        raise HTTPException(422, "The segmentation viewer supports at most 255 COCO categories")
    mapped = {int(item["id"]): item for item in categories if 1 <= int(item["id"]) <= 255}
    available = iter(value for value in range(1, 256) if value not in mapped)
    for item in sorted(categories, key=lambda item: int(item["id"])):
        if not 1 <= int(item["id"]) <= 255:
            mapped[next(available)] = item
    return dict(sorted(mapped.items()))


def rasterize_coco(document: dict, shape: tuple[int, ...]) -> np.ndarray:
    """Paint COCO annotations into a single-slice volume using viewer label IDs."""
    height, width = document["image"]["height"], document["image"]["width"]
    canvas = np.zeros((height, width), dtype=np.uint8)
    labels = {
        int(item["id"]): value for value, item in coco_categories(document["categories"]).items()
    }
    for annotation in document.get("annotations", []):
        label = labels[int(annotation["category_id"])]
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            encoded = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
        else:
            encoded = coco_mask.frPyObjects(
                {
                    "size": [height, width],
                    "counts": rle_counts(segmentation["counts"], height * width),
                },
                height,
                width,
            )
        canvas[coco_mask.decode(encoded).astype(bool)] = label
    volume = canvas[::-1, ::-1].T[..., None]
    if volume.shape != tuple(shape):
        raise HTTPException(409, "Annotation does not match the image dimensions")
    return volume


def annotation_volume(
    case: Case, annotation: AnnotationVersion | None, shape: tuple[int, ...]
) -> np.memmap | None:
    """Return the cached RAS label volume for an annotation version, or None."""
    if annotation is None or annotation.format == AnnotationFormat.NONE:
        return None
    # Previous COCO caches clamped category IDs, losing zero and merging IDs >255.
    suffix = "-coco-v2" if annotation.format == AnnotationFormat.COCO else ""
    relative = f"{viewer_dir(case)}/mask-{annotation.id}{suffix}.u8"
    if get_file_path(relative).is_file():
        return mapped(relative, shape, np.uint8)
    if annotation.format == AnnotationFormat.COCO:
        labels = rasterize_coco(read_json(get_file_path(annotation.annotation_path)), shape)
    else:
        try:
            labels, _, _ = read_nifti(get_file_path(annotation.annotation_path), np.uint8)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if labels.shape != tuple(shape):
            raise HTTPException(409, "Segmentation shape does not match the image")
    return store(relative, labels, np.uint8)


def work_dir(case: Case, user_id) -> str:
    return f"{case.dataset_id}/cases/{case.id}/work/{user_id}"


def work_state(case: Case, user_id) -> dict | None:
    path = get_file_path(f"{work_dir(case, user_id)}/meta.json")
    return json.loads(path.read_text()) if path.is_file() else None


def open_work(
    case: Case, user_id, shape: tuple[int, ...], base: np.memmap | None, state: dict | None = None
) -> np.memmap | None:
    """Open, or with ``state`` start, the reviewer's private editable label volume."""
    path = get_file_path(f"{work_dir(case, user_id)}/mask.u8")
    if path.is_file():
        return np.memmap(path, dtype=np.uint8, mode="r+", shape=tuple(shape), order=LAYOUT)
    if state is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = np.memmap(path, dtype=np.uint8, mode="w+", shape=tuple(shape), order=LAYOUT)
    if base is not None:
        buffer[:] = base
    buffer.flush()
    get_file_path(f"{work_dir(case, user_id)}/meta.json").write_text(json.dumps(state))
    return buffer


def discard_work(case: Case, user_id) -> None:
    shutil.rmtree(get_file_path(work_dir(case, user_id)), ignore_errors=True)


def face(volume, axis: int, index: int):
    """Basic indexing only: np.take copies through the fancy-index path and is ~300x slower."""
    if axis not in (0, 1, 2) or not 0 <= index < volume.shape[axis]:
        raise HTTPException(404, "Slice index out of range")
    if axis == 0:
        return volume[index]
    return volume[:, index] if axis == 1 else volume[:, :, index]


def extract(volume, axis: int, index: int) -> np.ndarray:
    """Return the display-oriented 2D slice: rows top-down, radiological columns."""
    return np.ascontiguousarray(face(volume, axis, index)[::-1, ::-1].T)


def insert(volume, axis: int, index: int, plane: np.ndarray) -> None:
    expected = face(volume, axis, index).shape[::-1]
    if plane.shape != expected:
        raise HTTPException(422, f"Slice must be {expected[1]} by {expected[0]} pixels")
    face(volume, axis, index)[:] = plane.T[::-1, ::-1]


def plane_spacing(spacing: list[float], axis: int) -> tuple[float, float]:
    """Return (row, column) millimetre spacing for the display orientation of an axis."""
    rest = [i for i in range(3) if i != axis]
    return spacing[rest[1]], spacing[rest[0]]


def png_bytes(plane: np.ndarray, level: float | None = None, width: float | None = None) -> bytes:
    if level is not None and width is not None:
        low = level - width / 2
        scaled = (plane.astype(np.float32) - low) * (255.0 / max(width, 1e-6))
        plane = np.clip(scaled, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    image = Image.fromarray(np.ascontiguousarray(plane, dtype=np.uint8), "L")
    image.save(buffer, "PNG", compress_level=1)
    return buffer.getvalue()


def nifti_payload(labels, case: Case, dataset: Dataset, meta: dict) -> bytes:
    """Serialize a RAS label volume as a NIfTI mask aligned with the source image."""
    original = np.ascontiguousarray(from_ras(labels, meta["ornt"]), dtype=np.uint8)
    header = None
    if dataset.image_format == ImageFormat.NIFTI:
        with mappable(get_file_path(case.image_path)) as usable:
            header = nib.load(usable).header.copy()
    mask = nib.Nifti1Image(original, np.asarray(meta["affine"], dtype=float), header)
    mask.set_data_dtype(np.uint8)
    mask.header.set_slope_inter(1, 0)
    return gzip.compress(mask.to_bytes(), compresslevel=1)


def coco_payload(labels, parent: dict, names: dict[int, str] | None = None) -> bytes:
    """Serialize a single-slice label volume as one COCO RLE annotation per category."""
    plane = np.asarray(labels[:, :, 0]).T[::-1, ::-1]
    annotations = []
    category_map = coco_categories(parent["categories"])
    for label in (int(value) for value in np.unique(plane) if value):
        if label not in category_map:
            raise HTTPException(422, "Use the existing COCO categories when painting this case")
        encoded = coco_mask.encode(np.asfortranarray((plane == label).astype(np.uint8)))
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": parent["image"]["id"],
                "category_id": category_map[label]["id"],
                "segmentation": {
                    "size": [int(x) for x in encoded["size"]],
                    "counts": encoded["counts"].decode("ascii"),
                },
                "bbox": [float(x) for x in coco_mask.toBbox(encoded)],
                "area": float(coco_mask.area(encoded)),
                "iscrowd": 0,
            }
        )
    category_names = {
        int(item["id"]): (names or {}).get(value) for value, item in category_map.items()
    }
    categories = [
        {**category, "name": category_names[int(category["id"])] or category.get("name", "")}
        for category in parent["categories"]
    ]
    document = {"image": parent["image"], "annotations": annotations, "categories": categories}
    return json.dumps(document).encode()

from pathlib import Path
from uuid import uuid4

import numpy as np
import pydicom
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnnotationFormat, AnnotationVersion, Case, Dataset, User
from app.services.annotation import coco_ids, nifti_metadata, read_json, validate_coco_case
from app.services.storage import copy_file, get_file_path, save_json, source_file


def case_directory(dataset: Dataset, case: Case) -> str:
    # Human identifiers from imported files never become filesystem paths.
    return f"{dataset.id}/cases/{case.id}"


def new_case(db: Session, dataset: Dataset, uid: str) -> Case:
    if not uid or len(uid) > 255:
        raise ValueError("Case identifier must contain 1–255 characters")
    if db.scalar(select(Case.id).where(Case.dataset_id == dataset.id, Case.case_uid == uid)):
        raise ValueError(f"Case already ingested: {uid}")
    return Case(id=uuid4(), dataset_id=dataset.id, case_uid=uid)


def add_original(db: Session, case: Case, path: str, format: AnnotationFormat, user: User):
    db.add(case)
    db.flush()
    db.add(
        AnnotationVersion(
            case_id=case.id, version=0, annotation_path=path, format=format, created_by=user.id
        )
    )
    db.flush()


def nifti_images(root: Path) -> list[Path]:
    images = sorted(p for p in (root / "images").glob("*") if p.name.endswith((".nii", ".nii.gz")))
    if not images:
        raise ValueError("No NIfTI images found in images/")
    return images


def ingest_nifti_dataset(db: Session, dataset: Dataset, root: Path, user: User) -> int:
    images = nifti_images(root)
    for image in images:
        image = source_file(root, f"images/{image.name}")
        mask = source_file(root, f"labels/{image.name}")
        suffix = ".nii.gz" if image.name.endswith(".nii.gz") else ".nii"
        case = new_case(db, dataset, image.name.removesuffix(suffix))
        base = case_directory(dataset, case)
        case.image_path = copy_file(db, image, f"{base}/image/image{suffix}")
        annotation_path = copy_file(db, mask, f"{base}/annotations/v0/segmentation{suffix}")
        try:
            case.metadata_json = nifti_metadata(
                get_file_path(case.image_path), get_file_path(annotation_path)
            )
        except ValueError as exc:
            raise ValueError(f"{image.name}: {exc}") from exc
        add_original(db, case, annotation_path, AnnotationFormat.NIFTI, user)
    return len(images)


def ingest_coco_dataset(db: Session, dataset: Dataset, root: Path, user: User) -> int:
    source = source_file(root, "annotations/instances.json")
    original_path = f"{dataset.id}/original/{uuid4()}/instances.json"
    copy_file(db, source, original_path)
    document = read_json(get_file_path(original_path))
    image_ids = coco_ids(document.get("images"), "images")
    coco_ids(document.get("categories"), "categories")
    coco_ids(document.get("annotations"), "annotations")
    if not image_ids:
        raise ValueError("COCO dataset must contain images")
    grouped = {image_id: [] for image_id in image_ids}
    for annotation in document["annotations"]:
        image_id = annotation.get("image_id")
        if type(image_id) is not int or image_id not in grouped:
            raise ValueError("COCO annotation references an unknown image")
        grouped[image_id].append(annotation)
    for image in document["images"]:
        filename = image.get("file_name")
        if not isinstance(filename, str):
            raise ValueError("COCO image requires file_name")
        source = source_file(root / "images", filename)
        if source.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("COCO images must be PNG or JPEG")
        case = new_case(db, dataset, Path(filename).with_suffix("").as_posix())
        base = case_directory(dataset, case)
        case.image_path = copy_file(db, source, f"{base}/image/image{source.suffix.lower()}")
        try:
            with Image.open(get_file_path(case.image_path)) as pixels:
                pixels.load()
                width, height = pixels.size
                image_format = pixels.format
                if image_format not in ("PNG", "JPEG"):
                    raise ValueError("COCO images must be PNG or JPEG")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Unreadable COCO image") from exc
        case_document = {
            "image": image,
            "annotations": grouped[image["id"]],
            "categories": document["categories"],
        }
        expected = {"id": image["id"], "width": width, "height": height}
        validate_coco_case(case_document, expected, document["categories"])
        path = save_json(db, case_document, f"{base}/annotations/v0/instances.json")
        case.metadata_json = {
            "width": width,
            "height": height,
            "coco_image_id": image["id"],
            "image_format": image_format,
            "categories": document["categories"],
            "original_coco_path": original_path,
        }
        add_original(db, case, path, AnnotationFormat.COCO, user)
    return len(image_ids)


def dicom_metadata(path: Path) -> dict:
    string_tags = (
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "Modality",
        "SeriesDescription",
    )
    integer_tags = ("Rows", "Columns", "InstanceNumber")
    vector_tags = {"PixelSpacing": 2, "ImagePositionPatient": 3, "ImageOrientationPatient": 6}
    try:
        header = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=[
                *string_tags,
                *integer_tags,
                *vector_tags,
                "SliceThickness",
                "NumberOfFrames",
            ],
        )
        result = {
            key: str(getattr(header, key)) if hasattr(header, key) else None for key in string_tags
        }
        if any(
            not result[key] for key in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
        ):
            raise ValueError("DICOM requires study, series, and SOP instance UIDs")
        if int(getattr(header, "NumberOfFrames", 1)) != 1:
            raise ValueError("Only single-frame DICOM series are supported")
        for key in integer_tags:
            result[key] = int(getattr(header, key)) if hasattr(header, key) else None
        if (
            not result["Rows"]
            or not result["Columns"]
            or min(result["Rows"], result["Columns"]) < 1
        ):
            raise ValueError("DICOM requires positive Rows and Columns")
        for key, length in vector_tags.items():
            values = [float(x) for x in getattr(header, key)] if hasattr(header, key) else None
            if values is not None and (len(values) != length or not np.isfinite(values).all()):
                raise ValueError(f"Invalid DICOM {key}")
            result[key] = values
        thickness = getattr(header, "SliceThickness", None)
        result["SliceThickness"] = float(thickness) if thickness is not None else None
        if result["SliceThickness"] is not None and not np.isfinite(result["SliceThickness"]):
            raise ValueError("Invalid DICOM SliceThickness")
        return result
    except (pydicom.errors.InvalidDicomError, OSError, TypeError, OverflowError) as exc:
        raise ValueError("Unreadable DICOM metadata") from exc


def sort_dicom_slices(slices: list[dict]) -> str:
    orientation = slices[0]["ImageOrientationPatient"]
    if orientation and all(
        s["ImagePositionPatient"] is not None
        and s["ImageOrientationPatient"] is not None
        and np.allclose(s["ImageOrientationPatient"], orientation, atol=1e-4)
        for s in slices
    ):
        normal = np.cross(orientation[:3], orientation[3:])
        if np.linalg.norm(normal) > 1e-8:
            slices.sort(
                key=lambda s: (
                    float(np.dot(s["ImagePositionPatient"], normal)),
                    s["SOPInstanceUID"],
                )
            )
            return "ImagePositionPatient"
    slices.sort(
        key=lambda s: (s["InstanceNumber"] is None, s["InstanceNumber"] or 0, s["SOPInstanceUID"])
    )
    return "InstanceNumber"


def ingest_dicom_dataset(db: Session, dataset: Dataset, root: Path, user: User) -> int:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".dcm")
    if not paths:
        raise ValueError("No .dcm files found")
    groups: dict[str, list[dict]] = {}
    for path in paths:
        path = source_file(root, str(path.relative_to(root)))
        metadata = dicom_metadata(path)
        metadata["source"] = path
        groups.setdefault(metadata["SeriesInstanceUID"], []).append(metadata)
    for uid, slices in groups.items():
        if len({s["SOPInstanceUID"] for s in slices}) != len(slices):
            raise ValueError("Duplicate SOPInstanceUID in DICOM series")
        if len({(s["StudyInstanceUID"], s["Rows"], s["Columns"]) for s in slices}) != 1:
            raise ValueError("Inconsistent study or image dimensions within DICOM series")
        sort_method = sort_dicom_slices(slices)
        case = new_case(db, dataset, uid)
        base = case_directory(dataset, case)
        case.image_path = f"{base}/image"
        for index, item in enumerate(slices):
            item["path"] = copy_file(db, item.pop("source"), f"{base}/image/{index:06d}.dcm")
        first = slices[0]
        case.metadata_json = {
            "StudyInstanceUID": first["StudyInstanceUID"],
            "SeriesInstanceUID": uid,
            "Modality": first["Modality"],
            "SeriesDescription": first["SeriesDescription"],
            "shape": [first["Rows"], first["Columns"], len(slices)],
            "sort_method": sort_method,
            "slices": slices,
        }
        db.add(case)
        db.flush()
    return len(groups)

import io
import json

import nibabel as nib
import numpy as np
import pytest
from conftest import create_dataset, ingest
from PIL import Image
from pycocotools import mask as coco_mask
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
from test_coco import coco_input

from app.services.volume import extract, insert

# An oblique, KiTS-style orientation: the array axes are (I, P, L), not (R, A, S).
KITS_AFFINE = np.array(
    [[0.0, 0.0, -0.8, 0.0], [0.0, -0.8, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def nifti_case(root, *, labelled=True, shape=(6, 8, 10), affine=KITS_AFFINE):
    (root / "images").mkdir(parents=True)
    image = np.arange(int(np.prod(shape)), dtype=np.int16).reshape(shape)
    nib.save(nib.Nifti1Image(image, affine), root / "images/case_0001.nii.gz")
    if labelled:
        (root / "labels").mkdir()
        mask = np.zeros(shape, dtype=np.uint8)
        mask[1:3, 2:4, 3:5] = 2
        nib.save(nib.Nifti1Image(mask, affine), root / "labels/case_0001.nii.gz")
    return image


def open_case(client, users, root, *, labelled=True):
    dataset_id = create_dataset(client, users["admin"])
    assert ingest(client, users["admin"], dataset_id, root).status_code == 201
    cases = client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()
    case_id = cases[0]["id"]
    info = client.get(f"/api/v1/viewer/cases/{case_id}", headers=users["doctor"])
    assert info.status_code == 200, info.text
    assert info.json()["has_mask"] is labelled
    return case_id, info.json()


def png_plane(response) -> np.ndarray:
    assert response.status_code == 200, response.text
    with Image.open(io.BytesIO(response.content)) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def paint(client, users, case_id, axis, index, plane):
    payload = io.BytesIO()
    Image.fromarray(plane, "L").save(payload, "PNG")
    return client.post(
        f"/api/v1/viewer/cases/{case_id}/paint/{axis}/{index}",
        headers={**users["doctor"], "Content-Type": "image/png"},
        content=payload.getvalue(),
    )


def test_slice_extraction_is_reversible():
    volume = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    for axis in (0, 1, 2):
        for index in (0, volume.shape[axis] - 1):
            plane = extract(volume, axis, index)
            target = np.zeros_like(volume)
            insert(target, axis, index, plane)
            assert np.array_equal(extract(target, axis, index), plane)
            assert np.array_equal(np.take(target, index, axis), np.take(volume, index, axis))


@pytest.mark.parametrize("kind", ["polygon", "uncompressed", "compressed"])
@pytest.mark.parametrize("category_id", [1, 0, 300])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_coco_viewer_and_saved_mask_round_trip(env, kind, category_id, image_format):
    client, settings, users, _ = env
    root = settings.import_dir / "coco"
    expected = np.zeros((6, 8), dtype=np.uint8)
    expected[1:4, 1:4] = 1
    segmentation = None
    if kind == "compressed":
        segmentation = coco_mask.encode(np.asfortranarray(expected))
        segmentation["counts"] = segmentation["counts"].decode("ascii")
    elif kind == "uncompressed":
        segmentation = {"size": [6, 8], "counts": [7, 3, 3, 3, 3, 3, 26]}
    document = coco_input(root, segmentation)
    document["categories"][0]["id"] = category_id
    document["annotations"][0]["category_id"] = category_id
    if image_format == "JPEG":
        Image.new("RGB", (8, 6), "red").save(root / "images/image_0001.jpg")
        document["images"][0]["file_name"] = "image_0001.jpg"
    (root / "annotations/instances.json").write_text(json.dumps(document))
    dataset_id = create_dataset(client, users["admin"], "COCO")
    response = ingest(client, users["admin"], dataset_id, root, "COCO")
    assert response.status_code == 201, response.text
    case = client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()[0]
    base = f"/api/v1/viewer/cases/{case['id']}"
    info = client.get(base, headers=users["doctor"])
    assert info.status_code == 200, info.text
    assert info.json()["labels"] == [{"value": 1, "name": "tumor"}]
    image = client.get(f"{base}/slice/2/0", headers=users["doctor"])
    assert image.status_code == 200
    assert image.headers["content-type"] == f"image/{image_format.lower()}"
    with Image.open(io.BytesIO(image.content)) as pixels:
        assert pixels.size == (8, 6)
    original = client.get(case["annotation_url"], headers=users["doctor"]).content
    assert np.array_equal(
        png_plane(client.get(f"{base}/mask/2/0", headers=users["doctor"])), expected
    )
    expected[5, 7] = 1
    assert paint(client, users, case["id"], 2, 0, expected).status_code == 204
    saved = client.post(f"{base}/save", headers=users["doctor"])
    assert saved.status_code == 201, saved.text
    assert np.array_equal(
        png_plane(client.get(f"{base}/mask/2/0", headers=users["doctor"])), expected
    )
    exported = json.loads((settings.data_dir / saved.json()["annotation_path"]).read_text())
    assert exported["categories"] == document["categories"]
    assert exported["annotations"][0]["category_id"] == category_id
    assert client.get(case["annotation_url"], headers=users["doctor"]).content == original


def test_viewer_reports_ras_geometry(env):
    client, settings, users, _ = env
    nifti_case(settings.import_dir / "nifti")
    _, info = open_case(client, users, settings.import_dir / "nifti")
    # The source array is (6, 8, 10) in (I, P, L); RAS order makes that (10, 8, 6).
    assert info["shape"] == [10, 8, 6]
    assert info["spacing"] == [0.8, 0.8, 3.0]
    assert [axis["name"] for axis in info["axes"]] == ["Axial", "Coronal", "Sagittal"]
    axial = info["axes"][0]
    assert (axial["count"], axial["rows"], axial["columns"]) == (6, 8, 10)
    assert info["labels"] == [{"value": 2, "name": "Label 2"}]


def test_mask_slice_matches_the_segmentation(env):
    client, settings, users, _ = env
    nifti_case(settings.import_dir / "nifti")
    case_id, info = open_case(client, users, settings.import_dir / "nifti")
    labelled = [
        index
        for index in range(info["axes"][0]["count"])
        if png_plane(
            client.get(f"/api/v1/viewer/cases/{case_id}/mask/2/{index}", headers=users["doctor"])
        ).any()
    ]
    # mask[1:3, 2:4, 3:5] along a flipped L axis covers two axial slices.
    assert len(labelled) == 2
    plane = png_plane(
        client.get(f"/api/v1/viewer/cases/{case_id}/mask/2/{labelled[0]}", headers=users["doctor"])
    )
    assert plane.shape == (8, 10)
    assert set(np.unique(plane).tolist()) == {0, 2}
    assert int((plane == 2).sum()) == 2 * 2  # Two P voxels by two I voxels.


def test_painting_saves_an_aligned_nifti_version(env):
    client, settings, users, _ = env
    source = nifti_case(settings.import_dir / "nifti")
    case_id, info = open_case(client, users, settings.import_dir / "nifti")
    axial = info["axes"][0]
    plane = np.zeros((axial["rows"], axial["columns"]), dtype=np.uint8)
    plane[0, 0] = 1
    plane[3, 4] = 3
    assert paint(client, users, case_id, 2, 2, plane).status_code == 204

    refreshed = client.get(f"/api/v1/viewer/cases/{case_id}", headers=users["doctor"]).json()
    assert refreshed["has_draft"] is True
    assert np.array_equal(
        png_plane(client.get(f"/api/v1/viewer/cases/{case_id}/mask/2/2", headers=users["doctor"])),
        plane,
    )

    saved = client.post(f"/api/v1/viewer/cases/{case_id}/save", headers=users["doctor"])
    assert saved.status_code == 201, saved.text
    assert saved.json()["version"] == 1
    assert (
        client.get(f"/api/v1/viewer/cases/{case_id}", headers=users["doctor"]).json()["has_draft"]
        is False
    )

    stored = settings.data_dir / saved.json()["annotation_path"]
    mask = nib.load(stored)
    assert mask.shape == source.shape
    assert np.allclose(mask.affine, KITS_AFFINE)
    values = np.asanyarray(mask.dataobj)
    assert sorted(np.unique(values).tolist()) == [0, 1, 2, 3]
    # The edited slice survives a save/reload round trip in the viewer's own frame.
    assert np.array_equal(
        png_plane(client.get(f"/api/v1/viewer/cases/{case_id}/mask/2/2", headers=users["doctor"])),
        plane,
    )


def test_segmentation_can_start_from_an_image_without_labels(env):
    client, settings, users, _ = env
    nifti_case(settings.import_dir / "plain", labelled=False)
    case_id, info = open_case(client, users, settings.import_dir / "plain", labelled=False)
    assert info["editable"] is True
    assert info["current_annotation"] is None
    axial = info["axes"][0]
    plane = np.zeros((axial["rows"], axial["columns"]), dtype=np.uint8)
    plane[1:4, 1:4] = 1
    assert paint(client, users, case_id, 2, 1, plane).status_code == 204
    saved = client.post(f"/api/v1/viewer/cases/{case_id}/save", headers=users["doctor"])
    assert saved.status_code == 201, saved.text
    # v0 stays reserved for an original this case never had.
    assert saved.json()["version"] == 1
    assert saved.json()["parent_id"] is None
    values = np.asanyarray(nib.load(settings.data_dir / saved.json()["annotation_path"]).dataobj)
    assert int((values == 1).sum()) == 9


def test_discard_restores_the_stored_segmentation(env):
    client, settings, users, _ = env
    nifti_case(settings.import_dir / "nifti")
    case_id, info = open_case(client, users, settings.import_dir / "nifti")
    axial = info["axes"][0]
    before = png_plane(
        client.get(f"/api/v1/viewer/cases/{case_id}/mask/2/2", headers=users["doctor"])
    )
    plane = np.full((axial["rows"], axial["columns"]), 1, dtype=np.uint8)
    assert paint(client, users, case_id, 2, 2, plane).status_code == 204
    assert (
        client.post(f"/api/v1/viewer/cases/{case_id}/discard", headers=users["doctor"]).status_code
        == 204
    )
    assert np.array_equal(
        png_plane(client.get(f"/api/v1/viewer/cases/{case_id}/mask/2/2", headers=users["doctor"])),
        before,
    )
    assert (
        client.post(f"/api/v1/viewer/cases/{case_id}/save", headers=users["doctor"]).status_code
        == 409
    )


def test_label_names_are_saved_beside_the_mask(env):
    client, settings, users, _ = env
    nifti_case(settings.import_dir / "nifti")
    case_id, info = open_case(client, users, settings.import_dir / "nifti")
    assert info["labels"] == [{"value": 2, "name": "Label 2"}]

    named = client.post(
        f"/api/v1/viewer/cases/{case_id}/labels",
        headers=users["doctor"],
        json={"labels": [{"value": 1, "name": "Kidney"}, {"value": 2, "name": "Tumor"}]},
    )
    assert named.status_code == 200, named.text
    assert named.json()["labels"] == [
        {"value": 1, "name": "Kidney"},
        {"value": 2, "name": "Tumor"},
    ]
    reopened = client.get(f"/api/v1/viewer/cases/{case_id}", headers=users["doctor"]).json()
    assert [item["name"] for item in reopened["labels"]] == ["Kidney", "Tumor"]

    axial = info["axes"][0]
    plane = np.zeros((axial["rows"], axial["columns"]), dtype=np.uint8)
    plane[2, 2] = 1
    assert paint(client, users, case_id, 2, 1, plane).status_code == 204
    saved = client.post(f"/api/v1/viewer/cases/{case_id}/save", headers=users["doctor"])
    assert saved.status_code == 201, saved.text
    sidecar = client.get(
        f"/api/v1/files/annotations/{saved.json()['id']}/labels", headers=users["doctor"]
    )
    assert sidecar.status_code == 200
    assert sidecar.json()["labels"] == {"1": "Kidney", "2": "Tumor"}


def dicom_volume(root, slices=3, rows=4, columns=5):
    """A small axial CT series with real pixel data and patient geometry."""
    root.mkdir(parents=True)
    study, series = generate_uid(), generate_uid()
    for index in range(slices):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        frame = FileDataset(str(root), {}, file_meta=meta, preamble=bytes(128))
        frame.StudyInstanceUID, frame.SeriesInstanceUID = study, series
        frame.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        frame.SOPClassUID, frame.Modality = CTImageStorage, "CT"
        frame.Rows, frame.Columns = rows, columns
        frame.InstanceNumber = index + 1
        frame.PixelSpacing = [0.8, 0.8]
        frame.SliceThickness = 2
        frame.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        frame.ImagePositionPatient = [0, 0, index * 2]
        frame.RescaleSlope, frame.RescaleIntercept = 1, -1024
        frame.SamplesPerPixel, frame.PhotometricInterpretation = 1, "MONOCHROME2"
        frame.BitsAllocated, frame.BitsStored, frame.HighBit = 16, 16, 15
        frame.PixelRepresentation = 0
        frame.PixelData = np.full((rows, columns), 1024 + index * 100, np.uint16).tobytes()
        frame.save_as(root / f"{index}.dcm", enforce_file_format=True)
    return series


def test_dicom_series_is_viewable_and_segmentable(env):
    client, settings, users, _ = env
    dicom_volume(settings.import_dir / "dicom")
    created = client.post(
        "/api/v1/datasets",
        headers=users["admin"],
        json={
            "name": "DICOM",
            "dimension": "3D",
            "image_format": "DICOM",
            "annotation_format": "NIFTI",
        },
    )
    assert created.status_code == 201, created.text
    dataset_id = created.json()["id"]
    assert (
        ingest(
            client, users["admin"], dataset_id, settings.import_dir / "dicom", "DICOM"
        ).status_code
        == 201
    )
    case_id = client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()[0][
        "id"
    ]

    info = client.get(f"/api/v1/viewer/cases/{case_id}", headers=users["doctor"]).json()
    assert info["shape"] == [5, 4, 3]  # (columns, rows, slices) once in RAS order.
    assert info["spacing"] == [0.8, 0.8, 2.0]
    assert info["editable"] is True and info["has_mask"] is False
    axial = info["axes"][0]
    assert (axial["name"], axial["count"], axial["rows"], axial["columns"]) == ("Axial", 3, 4, 5)

    # Stored pixels 1024/1124/1224 rescale to 0/100/200 HU, which a window of
    # 400 centred on 0 maps to 127/191/255 in slice order.
    rendered = []
    for index in range(3):
        plane = png_plane(
            client.get(
                f"/api/v1/viewer/cases/{case_id}/slice/2/{index}?level=0&width=400",
                headers=users["doctor"],
            )
        )
        assert plane.shape == (4, 5)
        rendered.append(int(plane.max()))
    assert rendered == [127, 191, 255]

    plane = np.zeros((axial["rows"], axial["columns"]), dtype=np.uint8)
    plane[1, 1] = 1
    assert paint(client, users, case_id, 2, 1, plane).status_code == 204
    saved = client.post(f"/api/v1/viewer/cases/{case_id}/save", headers=users["doctor"])
    assert saved.status_code == 201, saved.text
    mask = nib.load(settings.data_dir / saved.json()["annotation_path"])
    assert mask.shape == (5, 4, 3)
    assert int((np.asanyarray(mask.dataobj) == 1).sum()) == 1

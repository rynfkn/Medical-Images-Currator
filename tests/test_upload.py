import io

import nibabel as nib
import numpy as np
from conftest import create_dataset


def nifti_bytes(shape=(4, 5, 6), value=0, dtype=np.int16) -> bytes:
    image = nib.Nifti1Image(np.full(shape, value, dtype=dtype), np.eye(4))
    return nib.Nifti1Image.to_bytes(image)


def upload(client, headers, dataset_id, files):
    return client.post(f"/api/v1/datasets/{dataset_id}/upload", headers=headers, files=files)


def test_browser_upload_ingests_and_cleans_up(env):
    client, settings, users, _ = env
    dataset_id = create_dataset(client, users["admin"])
    response = upload(
        client,
        users["admin"],
        dataset_id,
        [
            ("images", ("case_a.nii", nifti_bytes(), "application/octet-stream")),
            ("images", ("case_b.nii", nifti_bytes(), "application/octet-stream")),
            ("labels", ("case_a.nii", nifti_bytes(value=1, dtype=np.uint8), "application/gzip")),
        ],
    )
    assert response.status_code == 201, response.text
    assert response.json()["cases_ingested"] == 2
    cases = client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()
    annotated = {case["case_uid"]: case["current_annotation"] for case in cases}
    assert annotated["case_a"]["version"] == 0
    assert annotated["case_b"] is None  # Uploaded without a matching segmentation.
    # Staged uploads are temporary; only the managed copies survive.
    assert not list((settings.import_dir / "uploads").rglob("*.nii"))


def test_upload_rejects_unsafe_names_and_wrong_types(env):
    client, settings, users, _ = env
    dataset_id = create_dataset(client, users["admin"])
    traversal = upload(
        client,
        users["admin"],
        dataset_id,
        [("images", ("../escape.nii", nifti_bytes(), "application/octet-stream"))],
    )
    # The name is reduced to its leaf, so the file lands inside the staging directory.
    assert traversal.status_code == 201
    assert [
        case["case_uid"]
        for case in client.get(
            f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]
        ).json()
    ] == ["escape"]
    assert not (settings.import_dir / "escape.nii").exists()

    wrong = upload(
        client,
        users["admin"],
        dataset_id,
        [("images", ("notes.txt", b"not an image", "text/plain"))],
    )
    assert wrong.status_code == 422
    assert "notes.txt" in wrong.json()["detail"]


def test_upload_requires_an_administrator(env):
    client, _, users, _ = env
    dataset_id = create_dataset(client, users["admin"])
    response = upload(
        client,
        users["doctor"],
        dataset_id,
        [("images", ("case_a.nii", nifti_bytes(), "application/octet-stream"))],
    )
    assert response.status_code == 403


def test_upload_rolls_back_when_a_file_is_invalid(env):
    client, settings, users, _ = env
    dataset_id = create_dataset(client, users["admin"])
    response = upload(
        client,
        users["admin"],
        dataset_id,
        [
            ("images", ("good.nii", nifti_bytes(), "application/octet-stream")),
            ("images", ("broken.nii", b"not a NIfTI payload", "application/octet-stream")),
        ],
    )
    assert response.status_code == 422
    assert client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json() == []
    assert not [path for path in settings.data_dir.rglob("*") if path.is_file()]
    assert not list((settings.import_dir / "uploads").rglob("*.nii"))


def test_coco_upload_needs_the_annotation_file(env):
    client, _, users, _ = env
    dataset_id = create_dataset(client, users["admin"], "COCO")
    response = upload(
        client,
        users["admin"],
        dataset_id,
        [("images", ("frame.png", io.BytesIO(b"fake").read(), "image/png"))],
    )
    assert response.status_code == 422
    assert "instances.json" in response.json()["detail"]

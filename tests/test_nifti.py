import nibabel as nib
import numpy as np
from conftest import create_dataset, ingest


def nifti_input(root, mask_shape=(4, 5, 6), mask_value=1):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    nib.save(
        nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.int16), np.eye(4)),
        root / "images/case_0001.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(np.full(mask_shape, mask_value, dtype=np.float32), np.eye(4)),
        root / "labels/case_0001.nii.gz",
    )


def test_nifti_ingest(env):
    client, settings, users, _ = env
    root = settings.import_dir / "nifti"
    nifti_input(root)
    dataset_id = create_dataset(client, users["admin"])
    response = ingest(client, users["admin"], dataset_id, root)
    assert response.status_code == 201, response.text
    cases = client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()
    assert len(cases) == 1
    case = cases[0]
    assert case["metadata"]["shape"] == [4, 5, 6]
    assert case["metadata"]["labels"] == [1]
    assert case["current_annotation"]["version"] == 0
    original = settings.data_dir / case["current_annotation"]["annotation_path"]
    assert original.read_bytes() == (root / "labels/case_0001.nii.gz").read_bytes()
    assert ingest(client, users["admin"], dataset_id, root).status_code == 422
    assert original.exists()


def test_shape_mismatch_rolls_back(env):
    client, settings, users, _ = env
    root = settings.import_dir / "nifti"
    nifti_input(root, mask_shape=(2, 3, 4))
    dataset_id = create_dataset(client, users["admin"])
    response = ingest(client, users["admin"], dataset_id, root)
    assert response.status_code == 422
    assert client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["admin"]).json() == []
    assert not list(settings.data_dir.rglob("*.nii.gz"))


def test_fractional_labels_rejected(env):
    client, settings, users, _ = env
    root = settings.import_dir / "nifti"
    nifti_input(root, mask_value=1.5)
    dataset_id = create_dataset(client, users["admin"])
    assert ingest(client, users["admin"], dataset_id, root).status_code == 422


def test_later_ingestion_failure_rolls_back_all_cases(env):
    client, settings, users, _ = env
    root = settings.import_dir / "nifti"
    nifti_input(root)
    # The first case is valid; the later image has no corresponding label.
    (root / "images/case_0002.nii.gz").write_bytes((root / "images/case_0001.nii.gz").read_bytes())
    dataset_id = create_dataset(client, users["admin"])
    assert ingest(client, users["admin"], dataset_id, root).status_code == 422
    assert client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["admin"]).json() == []
    assert not list(settings.data_dir.rglob("*.nii.gz"))

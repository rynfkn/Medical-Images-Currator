import json
import shutil
from uuid import UUID, uuid4

import pytest
from conftest import create_dataset, ingest
from sqlalchemy import event, select
from test_coco import coco_input
from test_reviews import correction_file, setup_case
from test_viewer import nifti_case

from app.models import AnnotationVersion, Case, Dataset, Review


@pytest.mark.parametrize("project", [False, True])
def test_delete_removes_dependents_and_managed_files_only(env, project):
    client, settings, users, sessions = env
    case = setup_case(env)
    other = create_dataset(client, users["admin"])
    other_root = settings.import_dir / "other"
    nifti_case(other_root)
    assert ingest(client, users["admin"], other, other_root).status_code == 201
    original_source = settings.import_dir / "nifti/images/case_0001.nii.gz"
    source_bytes = original_source.read_bytes()
    base = f"/api/v1/cases/{case['id']}"
    for _ in range(2):
        assert (
            client.post(
                f"{base}/annotations", headers=users["doctor"], files=correction_file(settings)
            ).status_code
            == 201
        )
    review = client.post(f"{base}/reviews", headers=users["doctor"], json={"decision": "MODIFIED"})
    assert review.status_code == 201
    assert (
        client.post(
            f"/api/v1/reviews/{review.json()['id']}/submit", headers=users["doctor"]
        ).status_code
        == 200
    )
    # Label deletion creates both cached volumes and private work files.
    viewer = f"/api/v1/viewer/cases/{case['id']}"
    names = client.get(viewer, headers=users["doctor"]).json()["labels"]
    assert (
        client.delete(f"{viewer}/labels/{names[0]['value']}", headers=users["doctor"]).status_code
        == 200
    )
    target = f"/api/v1/datasets/{case['dataset_id']}" if project else base
    assert client.delete(target).status_code == 401
    assert client.delete(target, headers=users["doctor"]).status_code == 403
    assert client.get(base, headers=users["doctor"]).status_code == 200
    assert client.delete(target, headers=users["admin"]).status_code == 204
    assert client.delete(target, headers=users["admin"]).status_code == 404
    assert client.get(base, headers=users["doctor"]).status_code == 404
    assert client.get(case["annotation_url"], headers=users["doctor"]).status_code == 404
    case_id = UUID(case["id"])
    with sessions() as db:
        assert db.get(Case, case_id) is None
        assert not list(
            db.scalars(select(AnnotationVersion).where(AnnotationVersion.case_id == case_id))
        )
        assert not list(db.scalars(select(Review).where(Review.case_id == case_id)))
        assert (db.get(Dataset, UUID(case["dataset_id"])) is None) == project
        assert db.get(Dataset, UUID(other)) is not None
    assert not (settings.data_dir / case["dataset_id"] / "cases" / case["id"]).exists()
    assert (settings.data_dir / other).exists()
    assert original_source.read_bytes() == source_bytes


def test_delete_case_keeps_siblings_and_shared_coco_source(env):
    client, settings, users, _ = env
    root = settings.import_dir / "coco"
    document = coco_input(root)
    shutil.copyfile(root / "images/image_0001.png", root / "images/image_0002.png")
    document["images"].append({**document["images"][0], "id": 2, "file_name": "image_0002.png"})
    (root / "annotations/instances.json").write_text(json.dumps(document))
    dataset_id = create_dataset(client, users["admin"], "COCO")
    assert ingest(client, users["admin"], dataset_id, root, "COCO").status_code == 201
    cases = client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()
    assert len(cases) == 2
    assert (
        client.delete(f"/api/v1/cases/{cases[0]['id']}", headers=users["admin"]).status_code == 204
    )
    assert client.get(cases[1]["image_url"], headers=users["doctor"]).status_code == 200
    assert (settings.data_dir / cases[1]["metadata"]["original_coco_path"]).exists()
    assert (
        client.delete(f"/api/v1/datasets/{dataset_id}", headers=users["admin"]).status_code == 204
    )
    assert not (settings.data_dir / dataset_id).exists()


def test_failed_transaction_preserves_files_and_records(env):
    client, settings, users, sessions = env
    case = setup_case(env)
    original = settings.data_dir / case["current_annotation"]["annotation_path"]
    before = original.read_bytes()

    def fail_commit(session):
        raise RuntimeError("Simulated database failure")

    event.listen(sessions, "before_commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="Simulated database failure"):
            client.delete(f"/api/v1/datasets/{case['dataset_id']}", headers=users["admin"])
    finally:
        event.remove(sessions, "before_commit", fail_commit)
    assert original.read_bytes() == before
    assert client.get(f"/api/v1/cases/{case['id']}", headers=users["doctor"]).status_code == 200


def test_empty_project_and_missing_targets(env):
    client, _, users, _ = env
    dataset_id = create_dataset(client, users["admin"])
    assert (
        client.delete(f"/api/v1/datasets/{dataset_id}", headers=users["admin"]).status_code == 204
    )
    assert client.delete(f"/api/v1/datasets/{uuid4()}", headers=users["admin"]).status_code == 404
    assert client.delete(f"/api/v1/cases/{uuid4()}", headers=users["admin"]).status_code == 404

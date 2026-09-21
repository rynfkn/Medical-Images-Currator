import json

import numpy as np
import pytest
from conftest import create_dataset, ingest
from test_coco import coco_input
from test_viewer import nifti_case, open_case, paint, png_plane


@pytest.mark.parametrize("kind", ["NIFTI", "COCO"])
def test_delete_label_can_be_discarded_or_saved_without_changing_history(env, kind):
    client, settings, users, _ = env
    root = settings.import_dir / "source"
    if kind == "NIFTI":
        nifti_case(root)
        case_id, info = open_case(client, users, root)
    else:
        document = coco_input(root)
        # Category zero must remain stably mapped to viewer label one.
        document["categories"][0]["id"] = 0
        document["annotations"][0]["category_id"] = 0
        (root / "annotations/instances.json").write_text(json.dumps(document))
        dataset_id = create_dataset(client, users["admin"], kind)
        assert ingest(client, users["admin"], dataset_id, root, kind).status_code == 201
        case_id = client.get(
            f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]
        ).json()[0]["id"]
        info = client.get(f"/api/v1/viewer/cases/{case_id}", headers=users["doctor"]).json()
    base = f"/api/v1/viewer/cases/{case_id}"
    value = info["labels"][0]["value"]
    original_url = info["current_annotation"]["url"]
    original = client.get(original_url, headers=users["doctor"]).content
    before = [
        png_plane(client.get(f"{base}/mask/2/{index}", headers=users["doctor"]))
        for index in range(info["shape"][2])
    ]
    assert any(np.any(plane == value) for plane in before)
    assert client.delete(f"{base}/labels/{value}", headers=users["doctor"]).json() == {"labels": []}
    assert client.get(base, headers=users["doctor"]).json()["has_draft"]
    assert client.get(base, headers=users["doctor"]).json()["labels"] == []
    # A private draft must not affect another reviewer or the original file.
    assert client.get(base, headers=users["other"]).json()["labels"] == info["labels"]
    for index in range(info["shape"][2]):
        assert not png_plane(client.get(f"{base}/mask/2/{index}", headers=users["doctor"])).any()
    assert client.post(f"{base}/discard", headers=users["doctor"]).status_code == 204
    assert client.get(base, headers=users["doctor"]).json()["labels"] == info["labels"]
    for index, plane in enumerate(before):
        assert np.array_equal(
            png_plane(client.get(f"{base}/mask/2/{index}", headers=users["doctor"])), plane
        )
    assert client.delete(f"{base}/labels/{value}", headers=users["doctor"]).status_code == 200
    saved = client.post(f"{base}/save", headers=users["doctor"])
    assert saved.status_code == 201, saved.text
    assert saved.json()["version"] == 1
    assert client.get(base, headers=users["doctor"]).json()["labels"] == []
    assert not client.get(base, headers=users["doctor"]).json()["has_draft"]
    assert client.get(original_url, headers=users["doctor"]).content == original
    if kind == "COCO":
        document = client.get(saved.json()["url"], headers=users["doctor"]).json()
        assert document["annotations"] == []
        assert document["categories"][0]["id"] == 0
    else:
        sidecar = client.get(f"{saved.json()['url']}/labels", headers=users["doctor"]).json()
        assert sidecar["labels"] == {}


def test_label_deletion_preserves_other_labels_and_unsaved_paint(env):
    client, settings, users, _ = env
    nifti_case(settings.import_dir / "source")
    case_id, info = open_case(client, users, settings.import_dir / "source")
    base = f"/api/v1/viewer/cases/{case_id}"
    assert (
        client.post(
            f"{base}/labels",
            headers=users["doctor"],
            json={
                "labels": [
                    {"value": 2, "name": "Tumor"},
                    {"value": 3, "name": "Kidney"},
                ]
            },
        ).status_code
        == 200
    )
    plane = np.full((info["shape"][1], info["shape"][0]), 3, dtype=np.uint8)
    assert paint(client, users, case_id, 2, 0, plane).status_code == 204
    assert client.delete(f"{base}/labels/2", headers=users["doctor"]).json() == {
        "labels": [{"value": 3, "name": "Kidney"}]
    }
    assert np.array_equal(png_plane(client.get(f"{base}/mask/2/0", headers=users["doctor"])), plane)
    assert client.post(f"{base}/save", headers=users["doctor"]).status_code == 201
    assert np.array_equal(png_plane(client.get(f"{base}/mask/2/0", headers=users["doctor"])), plane)


def test_label_validation_and_review_lock(env):
    client, settings, users, _ = env
    nifti_case(settings.import_dir / "source")
    case_id, _ = open_case(client, users, settings.import_dir / "source")
    base = f"/api/v1/viewer/cases/{case_id}"
    for name in ("", "   ", "a" * 61):
        assert (
            client.post(
                f"{base}/labels",
                headers=users["doctor"],
                json={"labels": [{"value": 2, "name": name}]},
            ).status_code
            == 422
        )
    assert client.post(
        f"{base}/labels",
        headers=users["doctor"],
        json={"labels": [{"value": 2, "name": "  Tumor  "}]},
    ).json() == {"labels": [{"value": 2, "name": "Tumor"}]}
    assert client.delete(f"{base}/labels/0", headers=users["doctor"]).status_code == 422
    assert client.delete(f"{base}/labels/256", headers=users["doctor"]).status_code == 422
    assert client.delete(f"{base}/labels/99", headers=users["doctor"]).status_code == 404
    assert client.delete(f"{base}/labels/2").status_code == 401
    assert (
        client.post(
            f"/api/v1/cases/{case_id}/reviews",
            headers=users["other"],
            json={"decision": "APPROVED"},
        ).status_code
        == 201
    )
    assert client.delete(f"{base}/labels/2", headers=users["doctor"]).status_code == 409
    assert (
        client.post(
            f"{base}/labels",
            headers=users["doctor"],
            json={"labels": [{"value": 2, "name": "Kidney"}]},
        ).status_code
        == 409
    )

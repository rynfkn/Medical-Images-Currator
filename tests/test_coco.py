import json

import numpy as np
import pytest
from conftest import create_dataset, ingest
from PIL import Image
from pycocotools import mask as coco_mask


def coco_input(root, segmentation=None):
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    Image.new("RGB", (8, 6)).save(root / "images/image_0001.png")
    document = {
        "images": [{"id": 1, "file_name": "image_0001.png", "width": 8, "height": 6}],
        "categories": [{"id": 1, "name": "tumor"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": segmentation
                if segmentation is not None
                else [[1, 1, 4, 1, 4, 4, 1, 4]],
                "bbox": [1, 1, 3, 3],
                "area": 9,
                "iscrowd": 0,
            }
        ],
    }
    (root / "annotations/instances.json").write_text(json.dumps(document))
    return document


@pytest.mark.parametrize("kind", ["polygon", "uncompressed", "compressed"])
def test_coco_ingestion_and_correction(env, kind):
    client, settings, users, _ = env
    segmentation = None
    if kind == "uncompressed":
        segmentation = {"size": [6, 8], "counts": [0, 48]}
    if kind == "compressed":
        mask = np.zeros((6, 8), dtype=np.uint8, order="F")
        mask[1:4, 1:4] = 1
        segmentation = coco_mask.encode(mask)
        segmentation["counts"] = segmentation["counts"].decode()
    root = settings.import_dir / "coco"
    document = coco_input(root, segmentation)
    dataset_id = create_dataset(client, users["admin"], "COCO")
    response = ingest(client, users["admin"], dataset_id, root, "COCO")
    assert response.status_code == 201, response.text
    case = client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()[0]
    original = client.get(case["annotation_url"], headers=users["doctor"]).content
    assert json.loads(original)["annotations"] == document["annotations"]
    assert (settings.data_dir / case["metadata"]["original_coco_path"]).read_bytes() == (
        root / "annotations/instances.json"
    ).read_bytes()
    corrected = json.loads(original)
    corrected["annotations"][0]["segmentation"] = [[0, 0, 5, 0, 5, 5, 0, 5]]
    response = client.post(
        f"/api/v1/cases/{case['id']}/annotations",
        headers=users["doctor"],
        files={"file": ("corrected.json", json.dumps(corrected))},
    )
    assert response.status_code == 201, response.text
    assert response.json()["version"] == 1
    assert client.get(case["annotation_url"], headers=users["doctor"]).content == original
    corrected["annotations"][0]["category_id"] = 99
    assert (
        client.post(
            f"/api/v1/cases/{case['id']}/annotations",
            headers=users["doctor"],
            files={"file": ("bad.json", json.dumps(corrected))},
        ).status_code
        == 422
    )
    assert not list(settings.data_dir.rglob("v2/*.json"))


@pytest.mark.parametrize("problem", ["missing_image", "category", "polygon", "rle", "size", "path"])
def test_invalid_coco_rolls_back(env, problem):
    client, settings, users, _ = env
    root = settings.import_dir / "coco"
    document = coco_input(root)
    if problem == "missing_image":
        (root / "images/image_0001.png").unlink()
    elif problem == "category":
        document["annotations"][0]["category_id"] = 123
    elif problem == "polygon":
        document["annotations"][0]["segmentation"] = [[1, 2, 3]]
    elif problem == "rle":
        document["annotations"][0]["segmentation"] = {"size": [6, 8], "counts": [49]}
    elif problem == "size":
        document["images"][0]["width"] = 9
    else:
        document["images"][0]["file_name"] = "../../outside.png"
    (root / "annotations/instances.json").write_text(json.dumps(document))
    dataset_id = create_dataset(client, users["admin"], "COCO")
    response = ingest(client, users["admin"], dataset_id, root, "COCO")
    assert response.status_code == 422, response.text
    assert client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json() == []
    assert not [p for p in settings.data_dir.rglob("*") if p.is_file()]

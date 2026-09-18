import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_reviews import correction_file, setup_case

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="Requires PostgreSQL row locks; set TEST_DATABASE_URL to a disposable database",
)


def test_concurrent_corrections_get_distinct_versions(env):
    client, settings, users, _ = env
    case = setup_case(env)
    payload = correction_file(settings)

    def upload():
        return client.post(
            f"/api/v1/cases/{case['id']}/annotations", headers=users["doctor"], files=payload
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: upload(), range(2)))
    assert [r.status_code for r in results] == [201, 201], [r.text for r in results]
    assert sorted(r.json()["version"] for r in results) == [1, 2]
    versions = client.get(f"/api/v1/cases/{case['id']}/annotations", headers=users["doctor"]).json()
    assert versions[2]["parent_id"] == versions[1]["id"]


def test_concurrent_submission_cannot_submit_twice(env):
    client, _, users, _ = env
    case = setup_case(env)
    response = client.post(
        f"/api/v1/cases/{case['id']}/reviews",
        headers=users["doctor"],
        json={"decision": "APPROVED"},
    )
    review_id = response.json()["id"]

    def submit():
        return client.post(f"/api/v1/reviews/{review_id}/submit", headers=users["doctor"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: submit(), range(2)))
    assert sorted(r.status_code for r in results) == [200, 409]

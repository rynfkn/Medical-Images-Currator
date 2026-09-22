import pytest


def test_browser_preflight_allows_bearer_and_range_headers(env):
    client, _, _, _ = env
    response = client.options(
        "/api/v1/datasets",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,range",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "authorization" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in response.headers


def test_unlisted_browser_origin_is_rejected(env):
    client, _, _, _ = env
    response = client.options(
        "/api/v1/datasets",
        headers={
            "Origin": "https://unlisted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_cors_keeps_authentication_required(env):
    client, _, users, _ = env
    origin = {"Origin": "http://localhost:5173"}
    response = client.get("/api/v1/auth/me", headers=origin)
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == origin["Origin"]
    response = client.get("/api/v1/auth/me", headers={**origin, **users["doctor"]})
    assert response.status_code == 200
    assert response.json()["role"] == "REVIEWER"
    assert "Content-Range" in response.headers["access-control-expose-headers"]


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/datasets/00000000-0000-0000-0000-000000000000",
        "/api/v1/cases/00000000-0000-0000-0000-000000000000",
        "/api/v1/viewer/cases/00000000-0000-0000-0000-000000000000/labels/1",
    ],
)
def test_browser_can_preflight_delete(env, path):
    client, _, _, _ = env
    response = client.options(
        path,
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "DELETE",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "DELETE" in response.headers["access-control-allow-methods"]
    response = client.delete(path, headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

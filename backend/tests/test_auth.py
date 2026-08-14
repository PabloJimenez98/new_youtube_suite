from httpx import AsyncClient

from app.models.user import User


async def test_login_success(client: AsyncClient, test_user: User) -> None:
    response = await client.post(
        "/auth/login",
        data={"username": test_user.email, "password": "s3cret-pw"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


async def test_login_wrong_password(client: AsyncClient, test_user: User) -> None:
    response = await client.post(
        "/auth/login",
        data={"username": test_user.email, "password": "wrong-password"},
    )
    assert response.status_code == 401


async def test_login_unknown_user(client: AsyncClient) -> None:
    response = await client.post(
        "/auth/login",
        data={"username": "nobody@example.com", "password": "whatever"},
    )
    assert response.status_code == 401


async def test_protected_route_rejects_missing_token(client: AsyncClient) -> None:
    response = await client.get("/uploads")
    assert response.status_code == 401


async def test_protected_route_rejects_invalid_token(client: AsyncClient) -> None:
    response = await client.get(
        "/uploads", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


async def test_protected_route_accepts_valid_token(
    client: AsyncClient, test_user: User
) -> None:
    login_response = await client.post(
        "/auth/login",
        data={"username": test_user.email, "password": "s3cret-pw"},
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/uploads", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json() == []

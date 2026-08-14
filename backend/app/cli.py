"""One-off operational commands.

v1 is single-operator/not multi-tenant (see docs/decisions/v1-scope.md
and docs/decisions/0002-backend-skeleton.md), so there is deliberately
no public `/auth/register` endpoint. The one user account is created
via this script instead.

Usage:
    python -m app.cli create-user someone@example.com
"""

import asyncio
import getpass
import sys

from app.auth.security import hash_password
from app.database import async_session_factory
from app.models.user import User


async def create_user(email: str, password: str) -> None:
    async with async_session_factory() as db:
        user = User(email=email, password_hash=hash_password(password))
        db.add(user)
        await db.commit()
        print(f"Created user #{user.id} <{user.email}>")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "create-user":
        print("Usage: python -m app.cli create-user <email>", file=sys.stderr)
        raise SystemExit(1)
    email = sys.argv[2]
    password = getpass.getpass("Password: ")
    asyncio.run(create_user(email, password))


if __name__ == "__main__":
    main()

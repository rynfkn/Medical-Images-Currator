import argparse
import getpass

from sqlalchemy import select

from app.core.security import password_hasher
from app.db import SessionLocal
from app.models import Role, User


def main():
    parser = argparse.ArgumentParser(description="Create an admin or reviewer")
    parser.add_argument("username")
    parser.add_argument("--full-name", required=True)
    parser.add_argument("--role", choices=[r.value for r in Role], required=True)
    args = parser.parse_args()
    password = getpass.getpass("Password (at least 12 characters): ")
    if len(password) < 12 or password != getpass.getpass("Confirm password: "):
        parser.error("Passwords must match and contain at least 12 characters")
    if not args.username.strip() or len(args.username) > 100 or len(args.full_name) > 200:
        parser.error("Invalid username or full name")
    with SessionLocal.begin() as db:
        if db.scalar(select(User).where(User.username == args.username)):
            parser.error("Username already exists")
        db.add(
            User(
                username=args.username,
                full_name=args.full_name,
                role=Role(args.role),
                password_hash=password_hasher.hash(password),
            )
        )
    print(f"Created {args.role.lower()} {args.username}")


if __name__ == "__main__":
    main()

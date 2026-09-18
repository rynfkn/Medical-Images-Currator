from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    # Commit before sending the response. Files written in this transaction are
    # removed on failure; committed originals are never on this cleanup list.
    with SessionLocal() as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            for path in reversed(session.info.get("created_files", [])):
                path.unlink(missing_ok=True)
            raise


Db = Annotated[Session, Depends(get_db, scope="function")]

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.core.security import CurrentUser, dummy_hash, make_token, password_hasher
from app.db import Db
from app.models import User
from app.schemas import Token, UserOut

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=Token)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: Db):
    user = db.scalar(select(User).where(User.username == form.username))
    valid = password_hasher.verify(form.password, user.password_hash if user else dummy_hash)
    if not user or not valid:
        raise HTTPException(
            401, "Incorrect username or password", headers={"WWW-Authenticate": "Bearer"}
        )
    return Token(access_token=make_token(user))


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser):
    return user

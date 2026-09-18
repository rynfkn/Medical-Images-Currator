from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://curator:curator@localhost:5432/curator"
    jwt_secret: SecretStr = Field(min_length=32)
    access_token_minutes: int = Field(default=480, ge=1)
    data_dir: Path = Path("data/datasets")
    import_dir: Path = Path("data/import")
    max_upload_bytes: int = Field(default=1024 * 1024 * 1024, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

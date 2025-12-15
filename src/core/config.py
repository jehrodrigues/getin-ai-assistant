# src/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centraliza todas as variáveis de ambiente do projeto.
    Carrega automaticamente o arquivo .env na raiz do repositório.

    Use:
        from src.core.config import settings
        settings.api_key_getin
    """

    # === GETIN API ===
    getin_api_key: str
    getin_api_base_url: str  # exemplo: https://sandbox.getinapis.com/apis/v2
    getin_default_unit_id: str | None = None

    # === Database / vectordb ===
    host: str = "localhost"
    port: int = 5432
    db: str = "getin_rag"
    user: str = ""
    password: str = ""

    # === LLM Providers ===
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    together_api_key: str | None = None
    mistral_api_key: str | None = None
    deepseek_api_key: str | None = None

    # === Misc ===
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
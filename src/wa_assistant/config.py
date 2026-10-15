"""Runtime settings (environment, prefix ``WA_``) and the business configuration (TOML)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from wa_assistant.hours import OpeningHours
from wa_assistant.text import LANGUAGES, Lang

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    """Every setting can be given as an environment variable ``WA_<NAME>`` (or in ``.env``)."""

    model_config = SettingsConfigDict(
        env_prefix="WA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- runtime --------------------------------------------------------------------------
    env: Literal["dev", "prod"] = "prod"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Path("data")
    db_path: str = "var/wa_assistant.db"

    # --- WhatsApp Cloud API -----------------------------------------------------------------
    verify_token: SecretStr = SecretStr("")
    app_secret: SecretStr = SecretStr("")
    access_token: SecretStr = SecretStr("")
    phone_number_id: str = ""
    graph_version: str = "v23.0"
    graph_base_url: str = "https://graph.facebook.com"
    http_timeout_seconds: float = Field(default=10.0, gt=0)
    mark_as_read: bool = True

    # --- operations ----------------------------------------------------------------------------
    owner_phone: str = ""
    admin_token: SecretStr = SecretStr("")
    handoff_ttl_hours: float = Field(default=24.0, gt=0)
    rate_limit_per_minute: int = Field(default=8, ge=0)  # 0 disables the limit

    # --- answers ------------------------------------------------------------------------------
    llm_provider: Literal["auto", "anthropic", "offline"] = "auto"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Effort = "low"
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)
    llm_fallbacks: bool = True  # server-side refusal fallbacks ("fallbacks": "default")
    history_turns: int = Field(default=6, ge=0, le=50)
    retrieval_top_k: int = Field(default=3, ge=1, le=10)
    offline_min_score: float = Field(default=1.5, ge=0)

    # Read without the WA_ prefix so the official variable name works, from .env too.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    @property
    def owner_wa_id(self) -> str:
        """The owner's number as WhatsApp sends it (digits only, no ``+``)."""
        return "".join(ch for ch in self.owner_phone if ch.isdigit())


@dataclass(frozen=True, slots=True)
class Service:
    id: str
    names: dict[Lang, str]

    def name(self, lang: Lang) -> str:
        return self.names[lang]


@dataclass(frozen=True, slots=True)
class BusinessConfig:
    name: str
    default_language: Lang
    owner_language: Lang
    hours: OpeningHours
    appointment_minutes: int
    max_days_ahead: int
    services: tuple[Service, ...]

    @property
    def tz(self) -> ZoneInfo:
        return self.hours.tz


def _language(value: object, field: str) -> Lang:
    if value not in LANGUAGES:
        msg = f"{field} must be one of {LANGUAGES}, got {value!r}"
        raise ValueError(msg)
    return value


def load_business_config(path: Path) -> BusinessConfig:
    """Parse ``business.toml``; raises ``ValueError`` with a readable message when invalid."""
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    business = raw.get("business", {})
    try:
        tz = ZoneInfo(str(business.get("timezone", "UTC")))
    except ZoneInfoNotFoundError as exc:
        msg = f"unknown timezone {business.get('timezone')!r}"
        raise ValueError(msg) from exc
    services = tuple(
        Service(
            id=str(item["id"]),
            names={lang: str(item.get(lang) or item["es"]) for lang in LANGUAGES},
        )
        for item in raw.get("services", [])
    )
    if not services:
        msg = "business.toml must define at least one [[services]] entry"
        raise ValueError(msg)
    booking = raw.get("booking", {})
    return BusinessConfig(
        name=str(business.get("name", "Demo business")),
        default_language=_language(business.get("default_language", "es"), "default_language"),
        owner_language=_language(business.get("owner_language", "es"), "owner_language"),
        hours=OpeningHours.from_mapping(
            tz, {str(k): str(v) for k, v in raw.get("hours", {}).items()}
        ),
        appointment_minutes=int(booking.get("appointment_minutes", 30)),
        max_days_ahead=int(booking.get("max_days_ahead", 60)),
        services=services,
    )

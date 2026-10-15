"""Shared fixtures. No test touches the network or needs an API key."""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from wa_assistant.app import create_app
from wa_assistant.config import BusinessConfig, Settings, load_business_config
from wa_assistant.engine import Engine
from wa_assistant.knowledge import KnowledgeBase
from wa_assistant.llm import OfflineProvider
from wa_assistant.payloads import InboundMessage
from wa_assistant.security import compute_signature
from wa_assistant.storage import Repository
from wa_assistant.whatsapp import RecordingSender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TZ = ZoneInfo("America/Argentina/Buenos_Aires")

# Reserved fictional numbers (555-01xx), never real people.
CUSTOMER = "15550100001"
OTHER_CUSTOMER = "15550100002"
OWNER = "15550100099"

APP_SECRET = "test-app-secret"
VERIFY_TOKEN = "test-verify-token"
ADMIN_TOKEN = "test-admin-token"

_ids = itertools.count(1)


def local(year: int, month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    """A business-local (Buenos Aires) aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=TZ)


class FakeClock:
    """A controllable clock; starts on Wednesday 2026-10-14 10:00 in Buenos Aires."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = (start or local(2026, 10, 14, 10, 0)).astimezone(UTC)

    def __call__(self) -> datetime:
        return self.now

    def set(self, moment: datetime) -> None:
        self.now = moment.astimezone(UTC)

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "env": "prod",
        "data_dir": DATA_DIR,
        "db_path": str(tmp_path / "test.db"),
        "app_secret": SecretStr(APP_SECRET),
        "verify_token": SecretStr(VERIFY_TOKEN),
        "admin_token": SecretStr(ADMIN_TOKEN),
        "access_token": SecretStr("test-access-token"),
        "phone_number_id": "100000000000001",
        "owner_phone": f"+{OWNER}",
        "llm_provider": "offline",
        # Tests send many messages at one fixed instant; the limiter has its own tests.
        "rate_limit_per_minute": 30,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def text_message(text: str, wa_id: str = CUSTOMER, message_id: str | None = None) -> InboundMessage:
    return InboundMessage(message_id or f"wamid.test{next(_ids)}", wa_id, "text", text)


def say(engine: Engine, sender: RecordingSender, text: str, wa_id: str = CUSTOMER) -> list[str]:
    """Send ``text`` from ``wa_id`` and return the replies that ``wa_id`` received."""
    before = len(sender.texts_to(wa_id))
    engine.process([text_message(text, wa_id)])
    return sender.texts_to(wa_id)[before:]


def webhook_payload(
    messages: list[dict[str, Any]] | None = None, statuses: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "15550100000", "phone_number_id": "100000000000001"},
    }
    if messages is not None:
        value["contacts"] = [
            {"profile": {"name": "Test User"}, "wa_id": m["from"]} for m in messages
        ]
        value["messages"] = messages
    if statuses is not None:
        value["statuses"] = statuses
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "200000000000001", "changes": [{"field": "messages", "value": value}]}],
    }


def wa_text(text: str, wa_id: str = CUSTOMER, message_id: str | None = None) -> dict[str, Any]:
    return {
        "from": wa_id,
        "id": message_id or f"wamid.hook{next(_ids)}",
        "timestamp": "1760450400",
        "type": "text",
        "text": {"body": text},
    }


def signed(
    payload: dict[str, Any] | list[Any], secret: str = APP_SECRET
) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(payload).encode("utf-8")
    return body, {
        "X-Hub-Signature-256": compute_signature(body, secret),
        "Content-Type": "application/json",
    }


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must not pick up the developer's WA_* or ANTHROPIC_* variables."""
    for key in list(os.environ):
        if key.startswith(("WA_", "ANTHROPIC_")):
            monkeypatch.delenv(key)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture(scope="session")
def business() -> BusinessConfig:
    return load_business_config(DATA_DIR / "business.toml")


@pytest.fixture(scope="session")
def knowledge() -> KnowledgeBase:
    return KnowledgeBase.load(DATA_DIR / "knowledge")


@pytest.fixture
def sender() -> RecordingSender:
    return RecordingSender()


@pytest.fixture
def repo(settings: Settings) -> Iterator[Repository]:
    repository = Repository(settings.db_path)
    yield repository
    repository.close()


@pytest.fixture
def engine(
    settings: Settings,
    business: BusinessConfig,
    knowledge: KnowledgeBase,
    repo: Repository,
    sender: RecordingSender,
    clock: FakeClock,
) -> Engine:
    return Engine(
        settings=settings,
        business=business,
        repo=repo,
        knowledge=knowledge,
        llm=OfflineProvider(min_score=settings.offline_min_score),
        sender=sender,
        clock=clock,
    )


@pytest.fixture
def client(settings: Settings, sender: RecordingSender, clock: FakeClock) -> Iterator[TestClient]:
    app = create_app(settings, sender=sender, llm=OfflineProvider(), clock=clock)
    with TestClient(app) as test_client:
        yield test_client

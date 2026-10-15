from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from tests.conftest import (
    ADMIN_TOKEN,
    CUSTOMER,
    VERIFY_TOKEN,
    FakeClock,
    make_settings,
    signed,
    wa_text,
    webhook_payload,
)
from wa_assistant.app import create_app
from wa_assistant.i18n import t
from wa_assistant.llm import OfflineProvider
from wa_assistant.security import bearer_token, compute_signature, tokens_match, verify_signature
from wa_assistant.whatsapp import RecordingSender

GREETING = t("greeting", "es", business="Clínica Dental Sonrisa")


def post(client: TestClient, payload: dict[str, Any], headers: dict[str, str] | None = None) -> int:
    body, signed_headers = signed(payload)
    response = client.post("/webhook", content=body, headers=headers or signed_headers)
    return response.status_code


# --- health and verification ---------------------------------------------------------------------------


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_verify_webhook_returns_the_challenge(client: TestClient) -> None:
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": VERIFY_TOKEN,
        "hub.challenge": "challenge-8765",
    }
    response = client.get("/webhook", params=params)
    assert response.status_code == 200
    assert response.text == "challenge-8765"
    assert response.headers["content-type"].startswith("text/plain")


@pytest.mark.parametrize(
    "params",
    [
        {"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "1"},
        {"hub.mode": "unsubscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "1"},
        {"hub.mode": "subscribe", "hub.challenge": "1"},
        {"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN},
        {},
    ],
)
def test_verify_webhook_rejects_bad_requests(client: TestClient, params: dict[str, str]) -> None:
    assert client.get("/webhook", params=params).status_code == 403


def test_verify_webhook_rejects_everything_without_a_configured_token(
    tmp_path: Path, sender: RecordingSender, clock: FakeClock
) -> None:
    app = create_app(
        make_settings(tmp_path, verify_token=SecretStr("")), sender=sender, clock=clock
    )
    params = {"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "1"}
    assert TestClient(app).get("/webhook", params=params).status_code == 403


# --- signature ------------------------------------------------------------------------------------------


def test_signed_message_is_answered_through_the_sender(
    client: TestClient, sender: RecordingSender
) -> None:
    status = post(client, webhook_payload([wa_text("¿Cuánto cuesta una limpieza?")]))
    assert status == 200
    replies = sender.texts_to(CUSTOMER)
    assert len(replies) == 1
    assert "ARS 45.000" in replies[0]
    assert replies[0].endswith("(Fuente: Precios y servicios)")
    assert len(sender.read) == 1


def test_invalid_signature_is_rejected(client: TestClient, sender: RecordingSender) -> None:
    body, headers = signed(webhook_payload([wa_text("hola")]), secret="not-the-secret")
    assert client.post("/webhook", content=body, headers=headers).status_code == 401
    assert sender.sent == []


def test_tampered_body_is_rejected(client: TestClient, sender: RecordingSender) -> None:
    body, headers = signed(webhook_payload([wa_text("hola")]))
    tampered = body.replace(b"hola", b"HOLA")
    assert client.post("/webhook", content=tampered, headers=headers).status_code == 401
    assert sender.sent == []


def test_missing_signature_is_rejected(client: TestClient, sender: RecordingSender) -> None:
    body, _ = signed(webhook_payload([wa_text("hola")]))
    response = client.post("/webhook", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 401
    assert sender.sent == []


def test_empty_app_secret_rejects_everything_outside_dev(
    tmp_path: Path, sender: RecordingSender, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        app = create_app(
            make_settings(tmp_path, app_secret=SecretStr("")), sender=sender, clock=clock
        )
    assert "every webhook POST will be rejected" in caplog.text
    body, headers = signed(webhook_payload([wa_text("hola")]), secret="")
    assert TestClient(app).post("/webhook", content=body, headers=headers).status_code == 401
    assert sender.sent == []


def test_dev_mode_without_app_secret_skips_the_check_with_a_warning(
    tmp_path: Path, sender: RecordingSender, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(tmp_path, app_secret=SecretStr(""), env="dev")
    with caplog.at_level(logging.WARNING):
        app = create_app(settings, sender=sender, llm=OfflineProvider(), clock=clock)
    assert "signatures are NOT verified" in caplog.text
    body, _ = signed(webhook_payload([wa_text("hola")]))
    with TestClient(app) as client:
        assert client.post("/webhook", content=body).status_code == 200
    assert sender.texts_to(CUSTOMER) == [GREETING]


# --- payload handling -------------------------------------------------------------------------------------


def test_duplicate_deliveries_are_ignored(client: TestClient, sender: RecordingSender) -> None:
    payload = webhook_payload([wa_text("hola", message_id="wamid.retry")])
    assert post(client, payload) == 200
    assert post(client, payload) == 200  # Meta retry of the same message
    assert sender.texts_to(CUSTOMER) == [GREETING]


def test_status_events_are_ignored(client: TestClient, sender: RecordingSender) -> None:
    statuses = [{"id": "wamid.out1", "status": "delivered", "recipient_id": CUSTOMER}]
    assert post(client, webhook_payload(statuses=statuses)) == 200
    assert sender.sent == []
    assert sender.read == []


def test_non_text_message_gets_a_text_only_reply(
    client: TestClient, sender: RecordingSender
) -> None:
    audio = {"from": CUSTOMER, "id": "wamid.audio", "type": "audio", "audio": {"id": "media1"}}
    assert post(client, webhook_payload([audio])) == 200
    assert sender.texts_to(CUSTOMER) == [t("unsupported_type", "es")]


@pytest.mark.parametrize("kind", ["button_reply", "list_reply"])
def test_interactive_replies_are_read_by_title(
    client: TestClient, sender: RecordingSender, kind: str
) -> None:
    interactive = {
        "from": CUSTOMER,
        "id": f"wamid.{kind}",
        "type": "interactive",
        "interactive": {"type": kind, kind: {"id": "opt-1", "title": "Quiero un turno"}},
    }
    assert post(client, webhook_payload([interactive])) == 200
    assert "¿Qué servicio necesitás?" in sender.texts_to(CUSTOMER)[0]


def test_invalid_json_is_a_bad_request(client: TestClient) -> None:
    body = b"{not json"
    headers = {"X-Hub-Signature-256": compute_signature(body, "test-app-secret")}
    assert client.post("/webhook", content=body, headers=headers).status_code == 400


def test_unexpected_shapes_are_acknowledged(client: TestClient, sender: RecordingSender) -> None:
    body, headers = signed({"entry": "nope"})
    response = client.post("/webhook", content=body, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    assert sender.sent == []


def test_other_objects_and_fields_are_ignored(client: TestClient, sender: RecordingSender) -> None:
    other_object = webhook_payload([wa_text("hola")]) | {"object": "page"}
    assert post(client, other_object) == 200
    other_field = webhook_payload([wa_text("hola")])
    other_field["entry"][0]["changes"][0]["field"] = "account_update"
    assert post(client, other_field) == 200
    assert sender.sent == []


def test_several_messages_in_one_delivery(client: TestClient, sender: RecordingSender) -> None:
    payload = webhook_payload([wa_text("hola"), wa_text("hello", wa_id="15550100002")])
    assert post(client, payload) == 200
    assert sender.texts_to(CUSTOMER) == [GREETING]
    assert len(sender.texts_to("15550100002")) == 1


def test_rate_limit_setting_is_applied(
    tmp_path: Path, sender: RecordingSender, clock: FakeClock
) -> None:
    settings = make_settings(tmp_path, rate_limit_per_minute=2)
    with TestClient(create_app(settings, sender=sender, clock=clock)) as client:
        for _ in range(4):
            assert post(client, webhook_payload([wa_text("hola")])) == 200
    assert sender.texts_to(CUSTOMER) == [GREETING, GREETING, t("rate_limited", "es")]


# --- admin --------------------------------------------------------------------------------------------------


def test_admin_requires_the_bearer_token(client: TestClient) -> None:
    assert client.get("/admin/bookings").status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.get("/admin/handoffs", headers=wrong).status_code == 401
    basic = {"Authorization": f"Basic {ADMIN_TOKEN}"}
    assert client.get("/admin/handoffs", headers=basic).status_code == 401
    assert client.get("/admin/bookings", headers=wrong).headers["www-authenticate"] == "Bearer"


def test_admin_lists_bookings_and_handoffs(client: TestClient) -> None:
    for step in ("turno", "1", "mañana", "10:00", "Ana García", "sí", "humano"):
        assert post(client, webhook_payload([wa_text(step)])) == 200
    auth = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    bookings = client.get("/admin/bookings", headers=auth).json()
    assert [(b["status"], b["date"], b["time"]) for b in bookings] == [
        ("requested", "2026-10-15", "10:00")
    ]
    handoffs = client.get("/admin/handoffs", headers=auth).json()
    assert handoffs[0]["wa_id"] == CUSTOMER
    assert client.get("/admin/bookings?limit=0", headers=auth).status_code == 422


def test_admin_is_hidden_without_a_configured_token(
    tmp_path: Path, sender: RecordingSender, clock: FakeClock
) -> None:
    app = create_app(make_settings(tmp_path, admin_token=SecretStr("")), sender=sender, clock=clock)
    response = TestClient(app).get("/admin/bookings", headers={"Authorization": "Bearer "})
    assert response.status_code == 404


# --- app wiring ----------------------------------------------------------------------------------------------


def test_docs_are_only_served_in_dev(tmp_path: Path, sender: RecordingSender) -> None:
    prod = TestClient(create_app(make_settings(tmp_path), sender=sender))
    assert prod.get("/docs").status_code == 404
    dev = TestClient(create_app(make_settings(tmp_path, env="dev"), sender=sender))
    assert dev.get("/docs").status_code == 200


def test_default_wiring_uses_the_graph_client(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
    assert type(app.state.engine.sender).__name__ == "WhatsAppClient"
    assert app.state.engine.llm.name == "offline"


# --- security primitives ----------------------------------------------------------------------------------------


def test_signature_helpers() -> None:
    body = b'{"a":1}'
    signature = compute_signature(body, "secret")
    assert signature.startswith("sha256=")
    assert verify_signature(body, signature, "secret")
    assert not verify_signature(body, signature, "other")
    assert not verify_signature(body, signature.removeprefix("sha256="), "secret")
    assert not verify_signature(body, None, "secret")
    assert not verify_signature(body, signature, "")


def test_token_helpers() -> None:
    assert bearer_token("Bearer abc") == "abc"
    assert bearer_token("bearer  abc ") == "abc"
    assert bearer_token("Basic abc") is None
    assert bearer_token("Bearer ") is None
    assert bearer_token(None) is None
    assert tokens_match("abc", "abc")
    assert not tokens_match("abd", "abc")
    assert not tokens_match(None, "abc")
    assert not tokens_match("", "")

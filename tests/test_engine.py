from __future__ import annotations

import logging
import threading
from dataclasses import replace

import pytest

from tests.conftest import (
    CUSTOMER,
    OTHER_CUSTOMER,
    OWNER,
    FakeClock,
    local,
    say,
    text_message,
)
from wa_assistant.config import Settings
from wa_assistant.engine import Engine, is_greeting, is_handoff_request
from wa_assistant.i18n import t
from wa_assistant.llm import AnswerRequest
from wa_assistant.payloads import InboundMessage
from wa_assistant.ratelimit import RateLimiter
from wa_assistant.storage import Repository
from wa_assistant.text import Lang
from wa_assistant.whatsapp import MAX_TEXT_LENGTH, RecordingSender, WhatsAppAPIError


class ScriptedLLM:
    """Records requests and returns a fixed reply."""

    def __init__(self, reply: str = "respuesta", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.requests: list[AnswerRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    def answer(self, request: AnswerRequest) -> str:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.reply


class FailingSender(RecordingSender):
    def send_text(self, to: str, body: str) -> None:
        raise WhatsAppAPIError(400, "Recipient phone number not in allowed list")

    def mark_as_read(self, message_id: str) -> None:
        raise WhatsAppAPIError(400, "Invalid message id")


# --- greeting, answers, language -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("hola", "es"),
        ("Buenas tardes", "es"),
        ("oi", "pt"),
        ("Olá, tudo bem?", "pt"),
        ("hello", "en"),
        ("menu", "es"),
    ],
)
def test_greeting_shows_the_localized_menu(
    engine: Engine, sender: RecordingSender, text: str, lang: Lang
) -> None:
    replies = say(engine, sender, text)
    assert replies == [t("greeting", lang, business="Clínica Dental Sonrisa")]


def test_greeting_detection() -> None:
    assert is_greeting("Hola!")
    assert is_greeting("hi there")
    assert not is_greeting("hola, ¿cuánto cuesta una limpieza?")
    assert not is_greeting("10:30")


def test_question_gets_a_grounded_answer_and_is_stored_as_history(
    engine: Engine, sender: RecordingSender, repo: Repository
) -> None:
    replies = say(engine, sender, "¿Cuánto cuesta una limpieza?")
    assert "ARS 45.000" in replies[0]
    assert replies[0].endswith("(Fuente: Precios y servicios)")
    assert repo.recent_messages(CUSTOMER, 10) == [
        ("user", "¿Cuánto cuesta una limpieza?"),
        ("assistant", replies[0]),
    ]


def test_unknown_question_offers_a_human(engine: Engine, sender: RecordingSender) -> None:
    assert say(engine, sender, "¿Venden cepillos eléctricos?") == [t("dont_know", "es")]


def test_history_and_language_reach_the_llm(engine: Engine, sender: RecordingSender) -> None:
    llm = ScriptedLLM("resposta")
    engine.llm = llm
    say(engine, sender, "Quanto custa um clareamento?")
    say(engine, sender, "E a limpeza?")
    second = llm.requests[1]
    assert second.lang == "pt"
    assert [turn.role for turn in second.history] == ["user", "assistant"]
    assert second.history[0].text == "Quanto custa um clareamento?"
    assert second.passages  # retrieval ran in Portuguese


def test_language_is_sticky(engine: Engine, sender: RecordingSender, repo: Repository) -> None:
    say(engine, sender, "Olá")
    say(engine, sender, "ok")
    assert repo.get_conversation(CUSTOMER).lang == "pt"


def test_long_answers_are_truncated_before_sending(engine: Engine, sender: RecordingSender) -> None:
    engine.llm = ScriptedLLM("a" * 6000)
    reply = say(engine, sender, "¿Cuánto cuesta una limpieza?")[0]
    assert len(reply) == MAX_TEXT_LENGTH


# --- reliability ------------------------------------------------------------------------------------


def test_duplicate_message_ids_are_processed_once(engine: Engine, sender: RecordingSender) -> None:
    message = text_message("hola", message_id="wamid.same")
    engine.process([message, message])
    engine.process([message])
    assert len(sender.texts_to(CUSTOMER)) == 1


def test_messages_are_marked_as_read(engine: Engine, sender: RecordingSender) -> None:
    engine.process([text_message("hola", message_id="wamid.read1")])
    assert sender.read == ["wamid.read1"]


def test_non_text_messages_get_a_localized_reply(engine: Engine, sender: RecordingSender) -> None:
    say(engine, sender, "Olá")
    engine.process([InboundMessage("wamid.img", CUSTOMER, "image", None)])
    assert sender.texts_to(CUSTOMER)[-1] == t("unsupported_type", "pt")


def test_rate_limit_replies_once_per_window(
    engine: Engine, sender: RecordingSender, clock: FakeClock
) -> None:
    engine.rate_limiter = RateLimiter(limit=3, window_seconds=60)
    for _ in range(3):
        assert say(engine, sender, "hola")
    assert say(engine, sender, "hola") == [t("rate_limited", "es")]
    assert say(engine, sender, "hola") == []
    assert say(engine, sender, "hola", wa_id=OTHER_CUSTOMER)  # other users are unaffected
    clock.advance(seconds=61)
    assert say(engine, sender, "hola") == [t("greeting", "es", business="Clínica Dental Sonrisa")]


def test_a_failing_message_does_not_block_the_batch(
    engine: Engine, sender: RecordingSender, caplog: pytest.LogCaptureFixture
) -> None:
    engine.llm = ScriptedLLM(error=RuntimeError("boom"))
    with caplog.at_level(logging.ERROR):
        engine.process([text_message("¿Cuánto cuesta una limpieza?"), text_message("hola")])
    assert sender.texts_to(CUSTOMER) == [t("greeting", "es", business="Clínica Dental Sonrisa")]
    assert "Failed to process message" in caplog.text


def test_send_failures_are_logged_not_raised(
    engine: Engine, repo: Repository, caplog: pytest.LogCaptureFixture
) -> None:
    engine.sender = FailingSender()
    with caplog.at_level(logging.WARNING):
        engine.process([text_message("humano")])
    assert "not in allowed list" in caplog.text
    assert "Could not mark message" in caplog.text
    assert repo.list_handoffs()[0]["owner_notified"] is False


def test_logs_mask_phones_and_never_contain_message_bodies(
    engine: Engine, sender: RecordingSender, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger="wa_assistant"):
        say(engine, sender, "¿Cuánto cuesta una limpieza? soy Ana Secreta")
        say(engine, sender, "humano")
    assert CUSTOMER not in caplog.text
    assert "+15*******01" in caplog.text
    assert "Secreta" not in caplog.text
    assert "limpieza" not in caplog.text


# --- human handoff -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "humano",
        "Quiero hablar con una persona",
        "atendente",
        "talk to a human please",
        "agente",
        # English had no word for "person": the simulator answered this one with
        # 'Would you like to talk to a person? Type "human".'
        "I want to talk to a person",
        "Can I speak to someone?",
        "a real representative",
        "Quiero hablar con alguien",
        "Quero falar com alguém",
    ],
)
def test_handoff_keywords(text: str) -> None:
    assert is_handoff_request(text)


def test_long_question_that_mentions_a_person_is_not_a_handoff() -> None:
    assert not is_handoff_request("¿Tiene que venir una persona adulta con mi hijo a la consulta?")
    assert not is_handoff_request("Does an adult person have to come with my son to the visit?")
    assert not is_handoff_request("Can someone else bring my child?")
    assert not is_handoff_request("¿Cuánto cuesta una limpieza?")


def test_handoff_silences_the_bot_until_the_ttl_expires(
    engine: Engine, sender: RecordingSender, repo: Repository, clock: FakeClock
) -> None:
    replies = say(engine, sender, "humano")
    assert replies == [t("handoff_started", "es")]  # 10:00 on a Wednesday: open
    handoffs = repo.list_handoffs()
    assert len(handoffs) == 1
    assert handoffs[0]["after_hours"] is False
    assert handoffs[0]["owner_notified"] is True
    alert = sender.texts_to(OWNER)[-1]
    assert f"+{CUSTOMER}" in alert
    assert "humano" in alert
    assert f"/bot on {CUSTOMER}" in alert

    clock.advance(hours=23)
    assert say(engine, sender, "¿Hola? ¿Hay alguien?") == []
    # The silenced customer's messages are relayed to the owner instead.
    forwarded = sender.texts_to(OWNER)[-1]
    assert "¿Hola? ¿Hay alguien?" in forwarded
    assert f"/reply {CUSTOMER}" in forwarded
    engine.process([InboundMessage("wamid.voice", CUSTOMER, "audio", None)])
    assert "[audio]" in sender.texts_to(OWNER)[-1]
    clock.advance(hours=2)  # WA_HANDOFF_TTL_HOURS = 24 by default
    assert say(engine, sender, "hola") == [t("greeting", "es", business="Clínica Dental Sonrisa")]
    assert repo.get_conversation(CUSTOMER).handoff_until is None


def test_owner_turns_the_bot_back_on(
    engine: Engine, sender: RecordingSender, repo: Repository
) -> None:
    say(engine, sender, "humano")
    assert say(engine, sender, "hola") == []
    owner_replies = say(engine, sender, f"/bot on +{CUSTOMER}", wa_id=OWNER)
    assert owner_replies == [t("owner_bot_on", "es", phone=CUSTOMER)]
    assert say(engine, sender, "hola") == [t("greeting", "es", business="Clínica Dental Sonrisa")]


def test_owner_replies_through_the_bot(engine: Engine, sender: RecordingSender) -> None:
    say(engine, sender, "humano")
    owner_replies = say(
        engine, sender, f"/reply {CUSTOMER} Hola Ana, te llamamos en 5 minutos.", wa_id=OWNER
    )
    assert owner_replies == [t("owner_reply_sent", "es", phone=CUSTOMER)]
    assert sender.texts_to(CUSTOMER)[-1] == "Hola Ana, te llamamos en 5 minutos."


class CustomerUnreachableSender(RecordingSender):
    def send_text(self, to: str, body: str) -> None:
        if to == CUSTOMER:
            raise WhatsAppAPIError(400, "Re-engagement message")
        super().send_text(to, body)


def test_owner_is_told_when_a_reply_is_not_accepted(engine: Engine) -> None:
    failing = CustomerUnreachableSender()
    engine.sender = failing
    engine.process([text_message(f"/reply {CUSTOMER} Hola", wa_id=OWNER)])
    assert failing.texts_to(OWNER) == [t("owner_reply_failed", "es", phone=CUSTOMER)]


@pytest.mark.parametrize("command", ["/help", "/bot off 123", "/reply", "/bot on nobody"])
def test_unknown_owner_commands_get_the_usage(
    engine: Engine, sender: RecordingSender, command: str
) -> None:
    assert say(engine, sender, command, wa_id=OWNER) == [t("owner_usage", "es")]


def test_only_the_owner_can_turn_the_bot_on(engine: Engine, sender: RecordingSender) -> None:
    say(engine, sender, "humano")
    say(engine, sender, f"/bot on {CUSTOMER}", wa_id=OTHER_CUSTOMER)
    assert say(engine, sender, "hola") == []


def test_owner_messages_without_a_slash_are_normal_messages(
    engine: Engine, sender: RecordingSender
) -> None:
    assert "asistente virtual" in say(engine, sender, "hola", wa_id=OWNER)[0]


def test_handoff_on_sunday_says_the_team_replies_on_the_next_business_day(
    engine: Engine, sender: RecordingSender, repo: Repository, clock: FakeClock
) -> None:
    clock.set(local(2026, 10, 18, 11, 0))  # Sunday
    reply = say(engine, sender, "quiero hablar con un agente")[0]
    assert reply.startswith(t("handoff_started", "es"))
    assert "próximo día hábil: lunes 19/10/2026 a las 09:00" in reply
    assert repo.list_handoffs()[0]["after_hours"] is True


def test_handoff_before_opening_says_today(
    engine: Engine, sender: RecordingSender, clock: FakeClock
) -> None:
    clock.set(local(2026, 10, 19, 7, 30))  # Monday, before 09:00
    reply = say(engine, sender, "atendente")[0]
    assert "hoje a partir das 09:00" in reply  # "atendente" is Portuguese


def test_handoff_after_saturday_closing_in_english(
    engine: Engine, sender: RecordingSender, clock: FakeClock
) -> None:
    clock.set(local(2026, 10, 17, 14, 0))  # Saturday afternoon
    say(engine, sender, "hello")
    reply = say(engine, sender, "I need a human")[0]
    assert "next business day: Monday 19/10/2026 at 09:00" in reply


def test_handoff_interrupts_a_booking_flow(
    engine: Engine, sender: RecordingSender, repo: Repository
) -> None:
    say(engine, sender, "turno")
    say(engine, sender, "humano")
    assert repo.get_conversation(CUSTOMER).flow_state is None


def test_no_owner_configured(engine: Engine, sender: RecordingSender, repo: Repository) -> None:
    engine.settings = Settings.model_construct(
        **{**engine.settings.model_dump(), "owner_phone": ""}
    )
    say(engine, sender, "humano")
    assert sender.texts_to(OWNER) == []
    assert repo.list_handoffs()[0]["owner_notified"] is False


def test_custom_handoff_ttl(engine: Engine, sender: RecordingSender, clock: FakeClock) -> None:
    engine.settings = Settings.model_construct(
        **{**engine.settings.model_dump(), "handoff_ttl_hours": 1.0}
    )
    say(engine, sender, "humano")
    assert "por 1 h" in sender.texts_to(OWNER)[-1]
    clock.advance(minutes=61)
    assert say(engine, sender, "hola")


def test_handoff_includes_the_profile_name(engine: Engine, sender: RecordingSender) -> None:
    message = replace(text_message("humano"), profile_name="Ana")
    engine.process([message])
    assert f"+{CUSTOMER} (Ana)" in sender.texts_to(OWNER)[-1]


def test_concurrent_customers_do_not_disturb_a_booking(
    engine: Engine, sender: RecordingSender, repo: Repository
) -> None:
    messages = ["turno", "1", "mañana", "10:00", "Ana García", "sí"]
    say(engine, sender, messages[0])
    threads = [
        threading.Thread(
            target=engine.process, args=([text_message("hola", wa_id=OTHER_CUSTOMER)],)
        )
        for _ in range(5)
    ]
    for thread in threads:
        thread.start()
    for step in messages[1:]:
        say(engine, sender, step)
    for thread in threads:
        thread.join()
    assert len(repo.list_bookings()) == 1
    assert len(sender.texts_to(OTHER_CUSTOMER)) == 5

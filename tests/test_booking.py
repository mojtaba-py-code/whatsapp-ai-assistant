from __future__ import annotations

from datetime import date, time

import pytest

from tests.conftest import CUSTOMER, OWNER, FakeClock, local, say
from wa_assistant.booking import (
    BookingState,
    clean_name,
    is_booking_intent,
    parse_date,
    parse_time,
)
from wa_assistant.engine import Engine
from wa_assistant.i18n import t
from wa_assistant.storage import Repository
from wa_assistant.whatsapp import RecordingSender

TODAY = date(2026, 10, 14)  # a Wednesday


# --- parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mañana", date(2026, 10, 15)),
        ("amanhã", date(2026, 10, 15)),
        ("tomorrow please", date(2026, 10, 15)),
        ("hoy", TODAY),
        ("20/10", date(2026, 10, 20)),
        ("el 5/11", date(2026, 11, 5)),
        ("20/10/2026", date(2026, 10, 20)),
        ("20-10-26", date(2026, 10, 20)),
        ("2026-10-20", date(2026, 10, 20)),
        ("01/10", date(2026, 10, 1)),  # already passed and next year is too far: stays past
        ("31/02", None),
        ("2026-13-01", None),
        ("pronto", None),
    ],
)
def test_parse_date(text: str, expected: date | None) -> None:
    assert parse_date(text, TODAY) == expected


def test_parse_date_rolls_dd_mm_into_next_year_in_december() -> None:
    assert parse_date("05/01", date(2026, 12, 20)) == date(2027, 1, 5)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10:30", time(10, 30)),
        ("9:05", time(9, 5)),
        ("10.30", time(10, 30)),
        ("10h30", time(10, 30)),
        ("10", time(10, 0)),
        ("10hs", time(10, 0)),
        ("a las 11", time(11, 0)),
        ("às 11", time(11, 0)),
        ("3pm", time(15, 0)),
        ("3:30 pm", time(15, 30)),
        ("12am", time(0, 0)),
        ("25:00", None),
        ("10:75", None),
        ("más tarde", None),
    ],
)
def test_parse_time(text: str, expected: time | None) -> None:
    assert parse_time(text) == expected


def test_clean_name() -> None:
    assert clean_name("  Ana   García ") == "Ana García"
    assert clean_name("O'Neil-Smith Jr.") == "O'Neil-Smith Jr."
    assert clean_name("1234") is None
    assert clean_name("A") is None
    assert clean_name("x" * 61) is None


@pytest.mark.parametrize(
    "text",
    [
        "turno",
        "Quiero un turno",
        "necesito una cita",
        "quero agendar",
        "Quero marcar uma consulta",
        "I want to book",
        "appointment",
    ],
)
def test_booking_intents(text: str) -> None:
    assert is_booking_intent(text)


@pytest.mark.parametrize(
    "text",
    [
        "¿Cuánto cuesta un turno?",
        "¿Cómo cancelo mi turno?",
        "What is your booking policy?",
        "¿Cuál es la política de citas?",
        "hola",
    ],
)
def test_questions_about_bookings_are_not_booking_intents(text: str) -> None:
    assert not is_booking_intent(text)


def test_state_round_trip_and_invalid_json() -> None:
    state = BookingState(step="time", service_id="cleaning", date="2026-10-15")
    assert BookingState.from_json(state.to_json()) == state
    assert BookingState.from_json(None) is None
    assert BookingState.from_json("not json") is None
    assert BookingState.from_json('{"unknown": 1}') is None
    assert BookingState.from_json('{"step": "dance"}') is None


# --- the flow, through the engine --------------------------------------------------------------------


def start_booking(engine: Engine, sender: RecordingSender) -> None:
    assert "1. Consulta y control general" in say(engine, sender, "Quiero un turno")[0]


def test_happy_path_saves_a_requested_booking(
    engine: Engine, sender: RecordingSender, repo: Repository
) -> None:
    replies = say(engine, sender, "turno")
    assert replies == [
        t("booking_ask_service", "es", options=engine.booking._options("es")),
    ]
    assert "Limpieza dental" in say(engine, sender, "2")[0]
    assert "jueves 15/10/2026" in say(engine, sender, "mañana")[0]
    assert say(engine, sender, "10:30") == [t("booking_ask_name", "es")]
    confirm = say(engine, sender, "Ana García")[0]
    assert "jueves 15/10/2026 a las 10:30" in confirm
    assert "Ana García" in confirm
    done = say(engine, sender, "Sí")
    assert done == [t("booking_saved", "es", id=1)]

    bookings = repo.list_bookings()
    assert len(bookings) == 1
    booking = bookings[0]
    assert booking["status"] == "requested"
    assert booking["service_id"] == "cleaning"
    assert (booking["date"], booking["time"]) == ("2026-10-15", "10:30")
    assert booking["patient_name"] == "Ana García"
    assert repo.get_conversation(CUSTOMER).flow_state is None
    # The owner gets an alert for every request.
    owner_alert = sender.texts_to(OWNER)[-1]
    assert "#1" in owner_alert
    assert CUSTOMER in owner_alert


def test_service_by_name_and_invalid_option(engine: Engine, sender: RecordingSender) -> None:
    start_booking(engine, sender)
    assert "1 al 5" in say(engine, sender, "9")[0]
    assert "1 al 5" in say(engine, sender, "algo")[0]
    assert "Blanqueamiento" in say(engine, sender, "blanqueamiento")[0]


def test_invalid_past_far_and_closed_dates(engine: Engine, sender: RecordingSender) -> None:
    start_booking(engine, sender)
    say(engine, sender, "1")
    assert say(engine, sender, "cuando puedan") == [t("booking_invalid_date", "es")]
    assert say(engine, sender, "01/10/2026") == [t("booking_past_date", "es")]
    assert "60 días" in say(engine, sender, "2027-03-01")[0]
    closed = say(engine, sender, "18/10")[0]  # Sunday
    assert closed.startswith("El domingo 18/10/2026 la clínica está cerrada")
    assert "lunes a viernes 09:00-19:00; sábado 09:00-13:00" in closed
    assert "¿A qué hora?" in say(engine, sender, "17/10")[0]  # Saturday is open


def test_invalid_closed_and_past_times(
    engine: Engine, sender: RecordingSender, clock: FakeClock
) -> None:
    start_booking(engine, sender)
    say(engine, sender, "1")
    asked = say(engine, sender, "hoy")[0]
    assert "el último a las 18:30" in asked
    assert say(engine, sender, "temprano") == [t("booking_invalid_time", "es")]
    assert "entre las 09:00 y las 18:30" in say(engine, sender, "19:00")[0]
    assert "entre las 09:00 y las 18:30" in say(engine, sender, "08:00")[0]
    past = say(engine, sender, "09:30")[0]  # it is 10:00 now
    assert past.startswith("Ese horario ya pasó")
    assert say(engine, sender, "11:00") == [t("booking_ask_name", "es")]
    assert say(engine, sender, "12345") == [t("booking_invalid_name", "es")]


def test_saturday_closes_early(engine: Engine, sender: RecordingSender) -> None:
    start_booking(engine, sender)
    say(engine, sender, "1")
    assert "el último a las 12:30" in say(engine, sender, "17/10")[0]
    assert "entre las 09:00 y las 12:30" in say(engine, sender, "12:45")[0]


def test_today_is_rejected_after_the_last_slot(
    engine: Engine, sender: RecordingSender, clock: FakeClock
) -> None:
    clock.set(local(2026, 10, 14, 18, 45))
    start_booking(engine, sender)
    say(engine, sender, "1")
    assert say(engine, sender, "hoy") == [t("booking_today_over", "es")]


def test_cancel_mid_flow(engine: Engine, sender: RecordingSender, repo: Repository) -> None:
    start_booking(engine, sender)
    say(engine, sender, "1")
    say(engine, sender, "mañana")
    assert say(engine, sender, "cancelar") == [t("booking_cancelled", "es")]
    assert repo.get_conversation(CUSTOMER).flow_state is None
    assert repo.list_bookings() == []
    # Out of the flow, messages are routed normally again.
    assert "asistente virtual" in say(engine, sender, "hola")[0]


def test_confirmation_no_discards_and_unclear_answer_asks_again(
    engine: Engine, sender: RecordingSender, repo: Repository
) -> None:
    start_booking(engine, sender)
    for step in ("1", "mañana", "10:00", "Juan Pérez"):
        say(engine, sender, step)
    assert say(engine, sender, "tal vez") == [t("booking_confirm_retry", "es")]
    assert say(engine, sender, "no") == [t("booking_declined", "es")]
    assert repo.list_bookings() == []


def test_portuguese_flow(engine: Engine, sender: RecordingSender, repo: Repository) -> None:
    assert "Qual serviço" in say(engine, sender, "Olá, quero marcar uma consulta")[0]
    assert "Limpeza dental" in say(engine, sender, "2")[0]
    assert "quinta-feira 15/10/2026" in say(engine, sender, "amanhã")[0]
    say(engine, sender, "15:00")
    assert "às 15:00" in say(engine, sender, "Maria Souza")[0]
    assert say(engine, sender, "sim") == [t("booking_saved", "pt", id=1)]
    assert repo.list_bookings()[0]["lang"] == "pt"


def test_english_flow(engine: Engine, sender: RecordingSender) -> None:
    assert "Which service" in say(engine, sender, "I want to book an appointment")[0]
    say(engine, sender, "1")
    assert "Thursday 15/10/2026" in say(engine, sender, "tomorrow")[0]
    say(engine, sender, "3pm")
    say(engine, sender, "John Smith")
    assert say(engine, sender, "yes") == [t("booking_saved", "en", id=1)]


def test_corrupted_flow_state_is_ignored(
    engine: Engine, sender: RecordingSender, repo: Repository, clock: FakeClock
) -> None:
    repo.set_flow_state(CUSTOMER, "{broken", clock())
    assert "asistente virtual" in say(engine, sender, "hola")[0]

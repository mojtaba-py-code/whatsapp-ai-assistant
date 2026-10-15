"""Opening hours, configuration, storage, rate limiting, logging helpers and the simulator."""

from __future__ import annotations

import io
from datetime import time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import DATA_DIR, PROJECT_ROOT, TZ, local
from wa_assistant.config import BusinessConfig, load_business_config
from wa_assistant.hours import DayHours, OpeningHours, parse_clock
from wa_assistant.logging_utils import mask_phone
from wa_assistant.ratelimit import RateDecision, RateLimiter
from wa_assistant.simulate import main as simulate
from wa_assistant.storage import MESSAGES_KEPT_PER_USER, PROCESSED_RETENTION, Repository

# --- opening hours -----------------------------------------------------------------------------------


def test_business_hours_from_the_demo_config(business: BusinessConfig) -> None:
    hours = business.hours
    assert hours.is_open(local(2026, 10, 14, 9, 0))  # Wednesday opening
    assert not hours.is_open(local(2026, 10, 14, 19, 0))  # closing time is exclusive
    assert hours.is_open(local(2026, 10, 17, 12, 59))  # Saturday morning
    assert not hours.is_open(local(2026, 10, 17, 13, 0))
    assert not hours.is_open(local(2026, 10, 18, 11, 0))  # Sunday
    assert hours.for_day(local(2026, 10, 18).date()) is None
    assert hours.for_day(local(2026, 10, 17).date()) == DayHours(time(9), time(13))


def test_is_open_converts_utc_to_buenos_aires(business: BusinessConfig) -> None:
    # 12:30 UTC is 09:30 in Buenos Aires (UTC-3).
    utc = local(2026, 10, 14, 9, 30).astimezone(ZoneInfo("UTC"))
    assert utc.hour == 12
    assert business.hours.is_open(utc)


def test_next_opening(business: BusinessConfig) -> None:
    hours = business.hours
    assert hours.next_opening(local(2026, 10, 17, 14, 0)) == local(2026, 10, 19, 9, 0)
    assert hours.next_opening(local(2026, 10, 19, 7, 0)) == local(2026, 10, 19, 9, 0)
    assert hours.next_opening(local(2026, 10, 14, 10, 0)) == local(2026, 10, 15, 9, 0)


def test_summary_groups_consecutive_days(business: BusinessConfig) -> None:
    assert business.hours.summary("es") == "lunes a viernes 09:00-19:00; sábado 09:00-13:00"
    assert business.hours.summary("en") == "Monday to Friday 09:00-19:00; Saturday 09:00-13:00"


def test_hours_validation() -> None:
    assert parse_clock("09:30") == time(9, 30)
    for bad in ("9", "nine:00", "9:5"):
        with pytest.raises(ValueError, match="expected HH:MM"):
            parse_clock(bad)
    with pytest.raises(ValueError, match="unknown weekday"):
        OpeningHours.from_mapping(TZ, {"monday": "09:00-10:00"})
    with pytest.raises(ValueError, match="expected HH:MM-HH:MM"):
        OpeningHours.from_mapping(TZ, {"mon": "09:00"})
    with pytest.raises(ValueError, match="close before they open"):
        OpeningHours.from_mapping(TZ, {"mon": "18:00-09:00"})
    with pytest.raises(ValueError, match="no opening hours"):
        OpeningHours.from_mapping(TZ, {}).next_opening(local(2026, 10, 14))


# --- business configuration ------------------------------------------------------------------------------


def test_demo_business_config(business: BusinessConfig) -> None:
    assert business.name == "Clínica Dental Sonrisa"
    assert str(business.tz) == "America/Argentina/Buenos_Aires"
    assert business.default_language == "es"
    assert [s.id for s in business.services] == [
        "checkup",
        "cleaning",
        "whitening",
        "orthodontics",
        "kids",
    ]
    assert business.services[1].name("pt") == "Limpeza dental"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "business.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown timezone"):
        load_business_config(_write(tmp_path, '[business]\ntimezone = "Mars/Olympus"\n'))
    with pytest.raises(ValueError, match="at least one"):
        load_business_config(_write(tmp_path, "[business]\nname = 'x'\n"))
    with pytest.raises(ValueError, match="default_language"):
        load_business_config(
            _write(
                tmp_path, '[business]\ndefault_language = "fr"\n[[services]]\nid = "a"\nes = "A"\n'
            )
        )


def test_minimal_config_falls_back_to_spanish_names(tmp_path: Path) -> None:
    config = load_business_config(
        _write(tmp_path, '[hours]\nmon = "09:00-12:00"\n[[services]]\nid = "a"\nes = "Consulta"\n')
    )
    assert config.services[0].name("en") == "Consulta"
    assert config.appointment_minutes == 30


# --- storage ------------------------------------------------------------------------------------------------


def test_claim_message_is_idempotent_and_old_ids_are_pruned(tmp_path: Path) -> None:
    repo = Repository(str(tmp_path / "db" / "nested.db"))
    now = local(2026, 10, 14)
    assert repo.claim_message("wamid.1", now)
    assert not repo.claim_message("wamid.1", now)
    later = now + PROCESSED_RETENTION + timedelta(minutes=1)
    assert repo.claim_message("wamid.2", later)  # prunes wamid.1
    assert repo.claim_message("wamid.1", later)
    repo.close()


def test_history_is_capped_per_user() -> None:
    repo = Repository(":memory:")
    now = local(2026, 10, 14)
    for index in range(MESSAGES_KEPT_PER_USER + 5):
        repo.add_message("1", "user", f"m{index}", now)
    repo.add_message("2", "user", "other", now)
    history = repo.recent_messages("1", 1000)
    assert len(history) == MESSAGES_KEPT_PER_USER
    assert history[-1] == ("user", f"m{MESSAGES_KEPT_PER_USER + 4}")
    assert repo.recent_messages("1", 0) == []
    assert repo.recent_messages("2", 5) == [("user", "other")]
    repo.close()


def test_wal_mode_is_enabled(tmp_path: Path) -> None:
    repo = Repository(str(tmp_path / "wal.db"))
    mode = repo._read("PRAGMA journal_mode", ())[0][0]
    assert mode == "wal"
    repo.close()


def test_conversation_round_trip(tmp_path: Path) -> None:
    repo = Repository(str(tmp_path / "c.db"))
    now = local(2026, 10, 14)
    assert repo.get_conversation("1").lang is None
    repo.set_language("1", "pt", now)
    repo.set_handoff("1", now + timedelta(hours=2), now)
    conversation = repo.get_conversation("1")
    assert conversation.lang == "pt"
    assert conversation.handoff_until == now + timedelta(hours=2)
    repo.set_handoff("1", None, now)
    assert repo.get_conversation("1").handoff_until is None
    repo.close()


# --- rate limiter ---------------------------------------------------------------------------------------------


def test_rate_limiter_windows() -> None:
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.check("a", 0) is RateDecision.ALLOW
    assert limiter.check("a", 1) is RateDecision.ALLOW
    assert limiter.check("a", 2) is RateDecision.NOTIFY
    assert limiter.check("a", 3) is RateDecision.DROP
    assert limiter.check("a", 61) is RateDecision.ALLOW


def test_rate_limiter_disabled_and_eviction() -> None:
    assert RateLimiter(limit=0).check("a", 0) is RateDecision.ALLOW
    limiter = RateLimiter(limit=1, window_seconds=60, max_keys=10)
    for index in range(25):
        assert limiter.check(str(index), float(index)) is RateDecision.ALLOW
    assert len(limiter._windows) <= 10


# --- logging -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phone", "masked"),
    [
        ("5491100000042", "+54*********42"),
        ("+1 555 010-0123", "+15*******23"),
        ("15550100001", "+15*******01"),
        ("1234", "+****"),
        ("", "<none>"),
    ],
)
def test_mask_phone(phone: str, masked: str) -> None:
    assert mask_phone(phone) == masked


# --- simulator --------------------------------------------------------------------------------------------------


def run_simulator(monkeypatch: pytest.MonkeyPatch, script: str, *args: str) -> str:
    monkeypatch.chdir(PROJECT_ROOT)
    out = io.StringIO()
    assert simulate(list(args), stdin=io.StringIO(script), stdout=out) == 0
    return out.getvalue()


def test_simulator_runs_a_spanish_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    script = "hola\n\n¿Cuánto cuesta una limpieza?\nhumano\n¿hola?\n/quit\nnot read\n"
    output = run_simulator(monkeypatch, script, "--now", "2026-10-14T10:00")
    assert "answers: offline" in output
    assert "you> ¿Cuánto cuesta una limpieza?" in output
    assert "(Fuente: Precios y servicios)" in output
    assert "bot> Listo, le aviso a una persona del equipo" in output
    assert "(no reply: the bot stays silent)" in output
    assert "not read" not in output


def test_simulator_language_and_phone_options(monkeypatch: pytest.MonkeyPatch) -> None:
    output = run_simulator(
        monkeypatch, "ok\n", "--lang", "pt", "--phone", "+1 555 010 0007", "--rate-limit", "5"
    )
    assert "customer: +15*******07" in output
    assert "bot> Não tenho essa informação" in output


def test_data_dir_is_the_project_data_folder() -> None:
    assert (DATA_DIR / "business.toml").is_file()

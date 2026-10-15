"""Deterministic appointment-request flow (a small state machine; the LLM is not involved).

Steps: service -> date -> time -> patient name -> confirmation. The result is a booking
*request* (status ``requested``) that the clinic confirms; the bot never promises a slot.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal, get_args

from wa_assistant.config import BusinessConfig, Service
from wa_assistant.i18n import format_day, format_when, t
from wa_assistant.text import Lang, fold, words

Step = Literal["service", "date", "time", "name", "confirm"]
_STEPS: tuple[Step, ...] = get_args(Step)

_TRIGGER_WORDS = frozenset(
    {"turno", "cita", "agendar", "reservar", "agendamento", "book", "appointment"}
)
_TRIGGER_PAIRS = (("marcar", "consulta"),)
# "¿Cuánto cuesta un turno?" or "¿cómo cancelo mi turno?" are questions for the knowledge
# base, not requests to start booking.
_QUESTION_PREFIXES = (
    "cancel", "reprogram", "remarc", "desmarc", "cambi", "precio", "preco", "price", "cost",
    "cuest", "cust", "polit", "policy",
)  # fmt: skip
_CANCEL_WORDS = frozenset({"cancelar", "cancel"})
_YES = frozenset(
    {"si", "sim", "yes", "y", "s", "ok", "okay", "dale", "confirmo", "confirmar", "claro"}
)
_NO = frozenset({"no", "nao", "n", "nope"})
_TOMORROW = frozenset({"manana", "amanha", "tomorrow"})
_TODAY = frozenset({"hoy", "hoje", "today"})

_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_DMY_DATE = re.compile(r"(?<!\d)(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{4}|\d{2}))?(?!\d)")
_CLOCK = re.compile(r"(?<!\d)(\d{1,2})[:.h](\d{2})(?!\d)\s*(am|pm)?")
_BARE_HOUR = re.compile(r"(?:a las |as |at )?(\d{1,2})\s*(?:h|hs|hrs|horas|hours)?\s*(am|pm)?")
_NAME_CHARS = frozenset(" '.-")


def is_booking_intent(text: str) -> bool:
    tokens = words(text)
    if any(token.startswith(_QUESTION_PREFIXES) for token in tokens):
        return False
    present = set(tokens)
    return bool(present & _TRIGGER_WORDS) or any(set(pair) <= present for pair in _TRIGGER_PAIRS)


def is_cancel(text: str) -> bool:
    return bool(set(words(text)) & _CANCEL_WORDS)


def parse_date(text: str, today: date) -> date | None:
    """``DD/MM``, ``DD/MM/YYYY``, ``YYYY-MM-DD``, "mañana"/"amanhã"/"tomorrow", "hoy"/"hoje"/"today".

    A ``DD/MM`` that already passed this year is read as next year's date only when that is
    within 60 days (e.g. "05/01" typed in December); otherwise it stays in the past.
    """
    tokens = set(words(text))
    if tokens & _TOMORROW:
        return today + timedelta(days=1)
    if tokens & _TODAY:
        return today
    try:
        if match := _ISO_DATE.search(text):
            return date(int(match[1]), int(match[2]), int(match[3]))
        if match := _DMY_DATE.search(text):
            day, month = int(match[1]), int(match[2])
            if match[3]:
                year = int(match[3])
                return date(year + 2000 if year < 100 else year, month, day)
            candidate = date(today.year, month, day)
            if candidate < today:
                next_year = date(today.year + 1, month, day)
                if next_year - today <= timedelta(days=60):
                    return next_year
            return candidate
    except ValueError:  # 31/02, month 13, ...
        return None
    return None


def parse_time(text: str) -> time | None:
    """``HH:MM`` (also ``HH.MM``, ``HHhMM``), a bare hour ("10", "10hs", "a las 10") and am/pm."""
    folded = fold(text).strip()
    match = _CLOCK.search(folded) or _BARE_HOUR.fullmatch(folded)
    if match is None:
        return None
    hour = int(match[1])
    minute = int(match[2]) if match.re is _CLOCK else 0
    suffix = match[3] if match.re is _CLOCK else match[2]
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def clean_name(text: str) -> str | None:
    name = " ".join(text.split())
    letters = sum(ch.isalpha() for ch in name)
    if not 2 <= len(name) <= 60 or letters < 2:
        return None
    if not all(ch.isalpha() or ch in _NAME_CHARS for ch in name):
        return None
    return name


@dataclass
class BookingState:
    step: Step = "service"
    service_id: str | None = None
    date: str | None = None  # ISO date
    time: str | None = None  # HH:MM
    name: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | None) -> BookingState | None:
        if not raw:
            return None
        try:
            data = json.loads(raw)
            state = cls(**data)
        except (ValueError, TypeError):
            return None
        return state if state.step in _STEPS else None


@dataclass(frozen=True, slots=True)
class BookingRequest:
    service: Service
    day: date
    at: time
    patient_name: str

    def when(self, lang: Lang) -> str:
        return format_when(datetime.combine(self.day, self.at), lang)


@dataclass(frozen=True, slots=True)
class FlowResult:
    reply: str
    state: BookingState | None  # None: the flow is over
    confirmed: BookingRequest | None = None


class BookingFlow:
    def __init__(self, business: BusinessConfig) -> None:
        self._business = business
        self._services = {service.id: service for service in business.services}

    # --- helpers ------------------------------------------------------------------------------
    def _options(self, lang: Lang) -> str:
        return "\n".join(
            f"{index}. {service.name(lang)}"
            for index, service in enumerate(self._business.services, start=1)
        )

    def _last_start(self, day: date) -> tuple[time, time] | None:
        hours = self._business.hours.for_day(day)
        if hours is None:
            return None
        closes = datetime.combine(day, hours.closes)
        last = (closes - timedelta(minutes=self._business.appointment_minutes)).time()
        return hours.opens, last

    def _match_service(self, text: str) -> Service | None:
        stripped = text.strip().rstrip(".)")
        if stripped.isdigit():
            index = int(stripped) - 1
            services = self._business.services
            return services[index] if 0 <= index < len(services) else None
        query = fold(stripped)
        if len(query) < 4:
            return None
        matches = {
            service.id
            for service in self._business.services
            for name in service.names.values()
            if query in fold(name) or fold(name) in query
        }
        return self._services[matches.pop()] if len(matches) == 1 else None

    # --- the flow -------------------------------------------------------------------------------
    def start(self, lang: Lang) -> FlowResult:
        return FlowResult(
            t("booking_ask_service", lang, options=self._options(lang)), BookingState()
        )

    def handle(self, state: BookingState, text: str, lang: Lang, now: datetime) -> FlowResult:
        """Advance the flow with the customer's ``text``; ``now`` must be timezone-aware."""
        if is_cancel(text):
            return FlowResult(t("booking_cancelled", lang), None)
        local_now = self._business.hours.local(now)
        match state.step:
            case "service":
                return self._on_service(state, text, lang, local_now)
            case "date":
                return self._on_date(state, text, lang, local_now)
            case "time":
                return self._on_time(state, text, lang, local_now)
            case "name":
                return self._on_name(state, text, lang)
            case "confirm":
                return self._on_confirm(state, text, lang)

    def _on_service(self, state: BookingState, text: str, lang: Lang, now: datetime) -> FlowResult:
        service = self._match_service(text)
        if service is None:
            reply = t(
                "booking_invalid_service",
                lang,
                count=len(self._business.services),
                options=self._options(lang),
            )
            return FlowResult(reply, state)
        state.service_id, state.step = service.id, "date"
        example = (now.date() + timedelta(days=1)).strftime("%d/%m")
        return FlowResult(
            t("booking_ask_date", lang, service=service.name(lang), example=example), state
        )

    def _on_date(self, state: BookingState, text: str, lang: Lang, now: datetime) -> FlowResult:
        today = now.date()
        day = parse_date(text, today)
        if day is None:
            return FlowResult(t("booking_invalid_date", lang), state)
        if day < today:
            return FlowResult(t("booking_past_date", lang), state)
        limit = today + timedelta(days=self._business.max_days_ahead)
        if day > limit:
            reply = t(
                "booking_too_far",
                lang,
                days=self._business.max_days_ahead,
                last=f"{limit:%d/%m/%Y}",
            )
            return FlowResult(reply, state)
        window = self._last_start(day)
        if window is None:
            reply = t(
                "booking_closed_day",
                lang,
                day=format_day(day, lang),
                hours=self._business.hours.summary(lang),
            )
            return FlowResult(reply, state)
        opens, last = window
        if day == today and now.time() >= last:
            return FlowResult(t("booking_today_over", lang), state)
        state.date, state.step = day.isoformat(), "time"
        reply = t(
            "booking_ask_time",
            lang,
            day=format_day(day, lang),
            open=f"{opens:%H:%M}",
            last=f"{last:%H:%M}",
        )
        return FlowResult(reply, state)

    def _on_time(self, state: BookingState, text: str, lang: Lang, now: datetime) -> FlowResult:
        at = parse_time(text)
        if at is None or state.date is None:
            return FlowResult(t("booking_invalid_time", lang), state)
        day = date.fromisoformat(state.date)
        window = self._last_start(day)
        if window is None:  # hours changed since the date was accepted: ask for a date again
            state.step, state.date = "date", None
            return FlowResult(t("booking_invalid_date", lang), state)
        opens, last = window
        values = {"day": format_day(day, lang), "open": f"{opens:%H:%M}", "last": f"{last:%H:%M}"}
        if not opens <= at <= last:
            return FlowResult(t("booking_time_closed", lang, **values), state)
        if day == now.date() and at <= now.time():
            return FlowResult(t("booking_time_past", lang, **values), state)
        state.time, state.step = f"{at:%H:%M}", "name"
        return FlowResult(t("booking_ask_name", lang), state)

    def _on_name(self, state: BookingState, text: str, lang: Lang) -> FlowResult:
        name = clean_name(text)
        if name is None:
            return FlowResult(t("booking_invalid_name", lang), state)
        state.name, state.step = name, "confirm"
        request = self._request(state)
        reply = t(
            "booking_confirm",
            lang,
            service=request.service.name(lang),
            when=request.when(lang),
            name=name,
        )
        return FlowResult(reply, state)

    def _on_confirm(self, state: BookingState, text: str, lang: Lang) -> FlowResult:
        tokens = words(text)
        first = tokens[0] if tokens else ""
        if first in _YES:
            return FlowResult("", None, confirmed=self._request(state))
        if first in _NO:
            return FlowResult(t("booking_declined", lang), None)
        return FlowResult(t("booking_confirm_retry", lang), state)

    def _request(self, state: BookingState) -> BookingRequest:
        if not (state.service_id and state.date and state.time and state.name):
            msg = "booking state is incomplete"
            raise ValueError(msg)
        return BookingRequest(
            service=self._services[state.service_id],
            day=date.fromisoformat(state.date),
            at=time.fromisoformat(state.time),
            patient_name=state.name,
        )

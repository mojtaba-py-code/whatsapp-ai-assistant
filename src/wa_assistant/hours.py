"""Opening hours in the business's own time zone (``zoneinfo`` + ``tzdata``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from wa_assistant.i18n import day_range
from wa_assistant.text import Lang

WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True, slots=True)
class DayHours:
    opens: time
    closes: time

    def label(self) -> str:
        return f"{self.opens:%H:%M}-{self.closes:%H:%M}"


def parse_clock(value: str) -> time:
    """``"09:00"`` -> ``time(9, 0)``; raises ``ValueError`` on anything else."""
    hours, sep, minutes = value.strip().partition(":")
    if not sep or not hours.isdigit() or not minutes.isdigit() or len(minutes) != 2:
        msg = f"invalid time {value!r}, expected HH:MM"
        raise ValueError(msg)
    return time(int(hours), int(minutes))


class OpeningHours:
    """Weekly opening hours; every ``datetime`` argument must be timezone-aware."""

    def __init__(self, tz: ZoneInfo, schedule: Mapping[int, DayHours]) -> None:
        self.tz = tz
        self._schedule = dict(schedule)

    @classmethod
    def from_mapping(cls, tz: ZoneInfo, raw: Mapping[str, str]) -> OpeningHours:
        """Build from ``{"mon": "09:00-19:00", ..., "sun": ""}`` (missing or empty = closed)."""
        unknown = set(raw) - set(WEEKDAY_KEYS)
        if unknown:
            msg = f"unknown weekday keys in [hours]: {sorted(unknown)}"
            raise ValueError(msg)
        schedule: dict[int, DayHours] = {}
        for index, key in enumerate(WEEKDAY_KEYS):
            value = raw.get(key, "").strip()
            if not value:
                continue
            opens_raw, sep, closes_raw = value.partition("-")
            if not sep:
                msg = f"invalid hours for {key}: {value!r}, expected HH:MM-HH:MM"
                raise ValueError(msg)
            day = DayHours(parse_clock(opens_raw), parse_clock(closes_raw))
            if day.opens >= day.closes:
                msg = f"hours for {key} close before they open: {value!r}"
                raise ValueError(msg)
            schedule[index] = day
        return cls(tz, schedule)

    def local(self, moment: datetime) -> datetime:
        return moment.astimezone(self.tz)

    def for_day(self, day: date) -> DayHours | None:
        return self._schedule.get(day.weekday())

    def is_open(self, moment: datetime) -> bool:
        local = self.local(moment)
        hours = self.for_day(local.date())
        return hours is not None and hours.opens <= local.time() < hours.closes

    def next_opening(self, moment: datetime) -> datetime:
        """The next moment (business-local) at which the business opens, after ``moment``."""
        local = self.local(moment)
        for offset in range(8):
            day = local.date() + timedelta(days=offset)
            hours = self.for_day(day)
            if hours is None:
                continue
            opening = datetime.combine(day, hours.opens, tzinfo=self.tz)
            if opening > local:
                return opening
        msg = "the business has no opening hours configured"
        raise ValueError(msg)

    def summary(self, lang: Lang) -> str:
        """``"lunes a viernes 09:00-19:00; sábado 09:00-13:00"`` - consecutive equal days grouped."""
        groups: list[tuple[int, int, DayHours]] = []
        for index in range(7):
            hours = self._schedule.get(index)
            if hours is None:
                continue
            if groups and groups[-1][1] == index - 1 and groups[-1][2] == hours:
                groups[-1] = (groups[-1][0], index, hours)
            else:
                groups.append((index, index, hours))
        return "; ".join(
            f"{day_range(first, last, lang)} {hours.label()}" for first, last, hours in groups
        )

"""Chat with the full engine in a terminal: no WhatsApp account and no API key needed.

    python -m wa_assistant.simulate --lang es
    python -m wa_assistant.simulate --now 2026-10-14T10:00 < conversation.txt

Messages go through the same engine as the webhook (language detection, booking flow,
handoff, retrieval, LLM provider); only the WhatsApp sender is replaced by the console.
The answer provider follows ``WA_LLM_PROVIDER`` (offline unless an Anthropic key is set).
"""

from __future__ import annotations

import argparse
import io
import sys
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import TextIO

from wa_assistant.config import Settings, load_business_config
from wa_assistant.engine import Clock, Engine, utc_now
from wa_assistant.knowledge import KnowledgeBase
from wa_assistant.llm import build_provider
from wa_assistant.logging_utils import mask_phone
from wa_assistant.payloads import InboundMessage
from wa_assistant.storage import Repository
from wa_assistant.text import LANGUAGES

DEFAULT_PHONE = "15550100001"  # a reserved fictional (555-01xx) number


class ConsoleSender:
    """Prints outbound WhatsApp messages instead of sending them."""

    def __init__(self, out: TextIO, customer: str) -> None:
        self._out = out
        self._customer = customer
        self.sent = 0

    def send_text(self, to: str, body: str) -> None:
        self.sent += 1
        prefix = "bot> " if to == self._customer else f"bot -> +{to}> "
        lines = body.splitlines() or [""]
        self._out.write(prefix + lines[0] + "\n")
        for line in lines[1:]:
            self._out.write((" " * len(prefix) + line if line else "") + "\n")

    def mark_as_read(self, message_id: str) -> None:
        """Nothing to mark in a terminal."""


def _utf8(*streams: TextIO) -> None:
    for stream in streams:
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def _fixed_clock(moment: datetime) -> Clock:
    def now() -> datetime:
        return moment

    return now


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wa_assistant.simulate",
        description="Chat with the WhatsApp assistant in the terminal (no WhatsApp, no keys).",
    )
    parser.add_argument(
        "--lang", choices=LANGUAGES, help="conversation language (default: detected)"
    )
    parser.add_argument("--phone", default=DEFAULT_PHONE, help="simulated customer number (digits)")
    parser.add_argument(
        "--now",
        help="fixed business-local time for the whole chat, e.g. 2026-10-14T10:00 (default: real time)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="messages per minute per user; 0 (default) turns the limit off for fast demos",
    )
    parser.add_argument(
        "--db", default=":memory:", help="SQLite database file (default: in memory)"
    )
    return parser


def main(
    argv: Sequence[str] | None = None, *, stdin: TextIO | None = None, stdout: TextIO | None = None
) -> int:
    args = _parser().parse_args(argv)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    _utf8(stdin, stdout)

    settings = Settings(db_path=args.db, rate_limit_per_minute=args.rate_limit, mark_as_read=False)
    business = load_business_config(settings.data_dir / "business.toml")
    clock: Clock = utc_now
    if args.now:
        fixed = datetime.fromisoformat(args.now)
        if fixed.tzinfo is None:
            fixed = fixed.replace(tzinfo=business.tz)
        clock = _fixed_clock(fixed)

    phone = "".join(ch for ch in args.phone if ch.isdigit()) or DEFAULT_PHONE
    repo = Repository(settings.db_path)
    llm = build_provider(settings, business.name)
    sender = ConsoleSender(stdout, phone)
    engine = Engine(
        settings=settings,
        business=business,
        repo=repo,
        knowledge=KnowledgeBase.load(settings.data_dir / "knowledge"),
        llm=llm,
        sender=sender,
        clock=clock,
    )
    if args.lang:
        repo.set_language(phone, args.lang, clock())

    stdout.write(
        f"{business.name} - WhatsApp assistant simulator\n"
        f"answers: {llm.name} | customer: {mask_phone(phone)} | type /quit to exit\n\n"
    )
    interactive = stdin.isatty()
    try:
        while True:
            if interactive:
                stdout.write("you> ")
                stdout.flush()
            line = stdin.readline()
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                break
            if not interactive:
                stdout.write(f"you> {text}\n")
            before = sender.sent
            engine.process([InboundMessage(f"sim-{uuid.uuid4().hex}", phone, "text", text)])
            if sender.sent == before:
                stdout.write("     (no reply: the bot stays silent)\n")
            stdout.write("\n")
            stdout.flush()
    finally:
        repo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

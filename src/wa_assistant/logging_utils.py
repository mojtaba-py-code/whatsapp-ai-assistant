"""Logging helpers. Phone numbers are masked; message bodies are never logged."""

from __future__ import annotations

import logging


def mask_phone(phone: str) -> str:
    """``"15550100001"`` -> ``"+15*******01"``: first two and last two digits only."""
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) <= 4:
        return "+" + "*" * len(digits) if digits else "<none>"
    return f"+{digits[:2]}{'*' * (len(digits) - 4)}{digits[-2:]}"


def configure_logging(level: str = "INFO") -> None:
    """Plain, single-line logs on stderr (only when the application has none configured)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("wa_assistant").setLevel(level)

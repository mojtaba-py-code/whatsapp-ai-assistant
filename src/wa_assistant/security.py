"""Webhook signature verification and constant-time token comparison."""

from __future__ import annotations

import hashlib
import hmac

SIGNATURE_PREFIX = "sha256="


def compute_signature(body: bytes, app_secret: str) -> str:
    """The ``X-Hub-Signature-256`` value Meta sends for ``body``."""
    digest = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return SIGNATURE_PREFIX + digest


def verify_signature(body: bytes, header: str | None, app_secret: str) -> bool:
    """``True`` only when ``header`` is the HMAC-SHA256 of the raw ``body`` with ``app_secret``."""
    if not app_secret or not header or not header.startswith(SIGNATURE_PREFIX):
        return False
    return hmac.compare_digest(compute_signature(body, app_secret), header.strip())


def tokens_match(supplied: str | None, expected: str) -> bool:
    """Constant-time comparison; an empty ``expected`` never matches."""
    if not expected or supplied is None:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def bearer_token(authorization: str | None) -> str | None:
    """The token of an ``Authorization: Bearer <token>`` header, or ``None``."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None

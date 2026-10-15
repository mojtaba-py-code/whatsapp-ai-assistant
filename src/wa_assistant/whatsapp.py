"""Outbound WhatsApp Cloud API client (Graph API) and the sender interface the engine uses."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx2

from wa_assistant.logging_utils import mask_phone

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 4096  # WhatsApp's limit for a text message body


def truncate_text(text: str, limit: int = MAX_TEXT_LENGTH) -> str:
    """Cut ``text`` to ``limit`` characters, marking the cut with an ellipsis."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class MessageSender(Protocol):
    """What the engine needs from WhatsApp; tests and the simulator inject fakes."""

    def send_text(self, to: str, body: str) -> None: ...

    def mark_as_read(self, message_id: str) -> None: ...


class WhatsAppAPIError(Exception):
    """The Graph API refused a request (4xx) or stayed unavailable after the retry."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(f"WhatsApp API error ({status_code}): {message}")
        self.status_code = status_code
        self.message = message


def _graph_error_message(response: httpx2.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(payload)[:200]


class WhatsAppClient:
    """``POST https://graph.facebook.com/{version}/{phone_number_id}/messages`` with a Bearer token.

    One retry on 5xx responses and transport errors (timeouts, refused or dropped
    connections); 4xx responses raise ``WhatsAppAPIError`` with the Graph error message.
    """

    def __init__(
        self,
        *,
        access_token: str,
        phone_number_id: str,
        graph_version: str = "v23.0",
        base_url: str = "https://graph.facebook.com",
        timeout: float = 10.0,
        retry_backoff: float = 0.5,
        http_client: httpx2.Client | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/{graph_version}/{phone_number_id}/messages"
        self._headers = {"Authorization": f"Bearer {access_token}"}
        self._retry_backoff = retry_backoff
        self._http = http_client or httpx2.Client(timeout=timeout)

    @property
    def url(self) -> str:
        return self._url

    def send_text(self, to: str, body: str) -> None:
        self._post(
            {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": truncate_text(body), "preview_url": False},
            }
        )
        logger.info("Sent a text message to %s", mask_phone(to))

    def mark_as_read(self, message_id: str) -> None:
        self._post({"messaging_product": "whatsapp", "status": "read", "message_id": message_id})

    def close(self) -> None:
        self._http.close()

    def _post(self, payload: dict[str, Any]) -> None:
        for attempt in (1, 2):
            try:
                response = self._http.post(self._url, json=payload, headers=self._headers)
            except httpx2.TransportError as exc:
                if attempt == 1:
                    logger.warning(
                        "WhatsApp API transport error (%s); retrying once", type(exc).__name__
                    )
                    time.sleep(self._retry_backoff)
                    continue
                raise WhatsAppAPIError(None, f"transport error: {type(exc).__name__}") from exc
            if response.status_code >= 500 and attempt == 1:
                logger.warning("WhatsApp API returned %s; retrying once", response.status_code)
                time.sleep(self._retry_backoff)
                continue
            if response.status_code >= 400:
                raise WhatsAppAPIError(response.status_code, _graph_error_message(response))
            return


@dataclass
class RecordingSender:
    """In-memory sender: records every outbound message (tests, simulator, local dev)."""

    sent: list[tuple[str, str]] = field(default_factory=list)
    read: list[str] = field(default_factory=list)

    def send_text(self, to: str, body: str) -> None:
        self.sent.append((to, body))

    def mark_as_read(self, message_id: str) -> None:
        self.read.append(message_id)

    def texts_to(self, wa_id: str) -> list[str]:
        return [body for to, body in self.sent if to == wa_id]

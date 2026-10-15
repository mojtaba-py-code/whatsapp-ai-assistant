"""Pydantic models for the WhatsApp Cloud API webhook payload (only the fields we use)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TextBody(_Model):
    body: str


class Reply(_Model):
    id: str = ""
    title: str


class Interactive(_Model):
    type: str
    button_reply: Reply | None = None
    list_reply: Reply | None = None


class WAMessage(_Model):
    id: str
    from_: str = Field(alias="from")
    type: str
    timestamp: str | None = None
    text: TextBody | None = None
    interactive: Interactive | None = None


class Profile(_Model):
    name: str | None = None


class Contact(_Model):
    wa_id: str
    profile: Profile | None = None


class ChangeValue(_Model):
    messaging_product: str | None = None
    contacts: list[Contact] = Field(default_factory=list)
    messages: list[WAMessage] = Field(default_factory=list)
    statuses: list[dict[str, Any]] = Field(default_factory=list)


class Change(_Model):
    field: str
    value: ChangeValue


class Entry(_Model):
    id: str | None = None
    changes: list[Change] = Field(default_factory=list)


class WebhookPayload(_Model):
    object_: str = Field(alias="object")
    entry: list[Entry] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """One customer message, reduced to what the engine needs."""

    id: str
    wa_id: str
    type: str
    text: str | None  # None for message types the bot cannot read (image, audio, ...)
    profile_name: str | None = None


def _message_text(message: WAMessage) -> str | None:
    if message.type == "text" and message.text is not None:
        return message.text.body
    if message.type == "interactive" and message.interactive is not None:
        reply = message.interactive.button_reply or message.interactive.list_reply
        return reply.title if reply is not None else None
    return None


def extract_messages(payload: WebhookPayload) -> tuple[list[InboundMessage], int]:
    """Messages to process and the number of ``statuses`` events that were ignored."""
    messages: list[InboundMessage] = []
    statuses = 0
    if payload.object_ != "whatsapp_business_account":
        return messages, statuses
    for entry in payload.entry:
        for change in entry.changes:
            if change.field != "messages":
                continue
            statuses += len(change.value.statuses)
            names = {
                c.wa_id: c.profile.name for c in change.value.contacts if c.profile is not None
            }
            messages.extend(
                InboundMessage(
                    id=message.id,
                    wa_id=message.from_,
                    type=message.type,
                    text=_message_text(message),
                    profile_name=names.get(message.from_),
                )
                for message in change.value.messages
            )
    return messages, statuses

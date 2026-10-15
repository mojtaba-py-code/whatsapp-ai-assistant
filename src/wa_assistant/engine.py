"""The conversation engine: routes every inbound WhatsApp message.

Order: idempotency -> owner commands -> handoff silence (messages relayed to the owner) ->
handoff request -> rate limit -> non-text messages -> booking flow in progress ->
booking request -> greeting/menu -> grounded answer (retrieval + LLM).
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from wa_assistant.booking import BookingFlow, BookingState, FlowResult, is_booking_intent
from wa_assistant.config import BusinessConfig, Settings
from wa_assistant.i18n import format_when, t
from wa_assistant.knowledge import KnowledgeBase
from wa_assistant.language import resolve_language
from wa_assistant.llm import AnswerRequest, LLMProvider, Turn
from wa_assistant.logging_utils import mask_phone
from wa_assistant.payloads import InboundMessage
from wa_assistant.ratelimit import RateDecision, RateLimiter
from wa_assistant.storage import BookingRecord, Conversation, Repository
from wa_assistant.text import Lang, words
from wa_assistant.whatsapp import MessageSender, WhatsAppAPIError, truncate_text

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]

HANDOFF_WORDS = frozenset(
    {
        "humano",
        "humana",
        "persona",
        "pessoa",
        "agente",
        "atendente",
        "human",
        "person",
        "agent",
        "representative",
        "operador",
        "operator",
    }
)
# A handoff keyword triggers on short messages ("humano", "quiero un agente") or when the
# message asks to talk to someone; longer questions that merely mention a "persona" do not.
_TALK_WORDS = frozenset(
    {
        "hablar", "falar", "talk", "speak", "comunicar", "comunicarme", "contactar", "quiero",
        "quero", "want", "need", "necesito", "preciso", "pasame", "passa", "transferir",
        "transfer", "real",
    }
)  # fmt: skip
_HANDOFF_MAX_WORDS = 6
# "Someone" only means a person from the team next to a verb such as "talk": "I want to
# talk to someone" is a handoff request, "can someone else bring my child?" is not.
_SOMEONE_WORDS = frozenset({"someone", "somebody", "alguien", "alguem"})
_GREETING_WORDS = frozenset(
    {
        "hola", "holi", "buenas", "buenos", "buen", "dia", "dias", "tarde", "tardes", "noche",
        "noches", "oi", "ola", "bom", "boa", "hello", "hi", "hey", "good", "morning",
        "afternoon", "evening", "menu", "inicio", "start", "ayuda", "ajuda", "help",
    }
)  # fmt: skip
_GREETING_FILLER = frozenset(
    {"como", "estas", "esta", "tudo", "bem", "vai", "how", "are", "you", "there"}
)
_OWNER_LAST_MESSAGE_CHARS = 300


def utc_now() -> datetime:
    return datetime.now(UTC)


def is_handoff_request(text: str) -> bool:
    tokens = words(text)
    found = set(tokens)
    if found & _SOMEONE_WORDS and found & _TALK_WORDS:
        return True
    if not found & HANDOFF_WORDS:
        return False
    return len(tokens) <= _HANDOFF_MAX_WORDS or bool(found & _TALK_WORDS)


def is_greeting(text: str) -> bool:
    tokens = set(words(text))
    return bool(tokens & _GREETING_WORDS) and tokens <= _GREETING_WORDS | _GREETING_FILLER


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


class Engine:
    def __init__(
        self,
        *,
        settings: Settings,
        business: BusinessConfig,
        repo: Repository,
        knowledge: KnowledgeBase,
        llm: LLMProvider,
        sender: MessageSender,
        clock: Clock = utc_now,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.settings = settings
        self.business = business
        self.repo = repo
        self.knowledge = knowledge
        self.llm = llm
        self.sender = sender
        self.clock = clock
        self.rate_limiter = rate_limiter or RateLimiter(settings.rate_limit_per_minute, 60.0)
        self.booking = BookingFlow(business)
        # Messages from one customer are handled one at a time (webhook batches run in a
        # thread pool, and the booking flow and rate limiter are per-customer state).
        self._customer_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    # --- entry points ------------------------------------------------------------------------------
    def process(self, messages: Sequence[InboundMessage]) -> None:
        """Handle a webhook batch; one failing message never blocks the others."""
        for message in messages:
            try:
                self.handle(message)
            except Exception:  # background task: log and keep going
                logger.exception(
                    "Failed to process message %s from %s", message.id, mask_phone(message.wa_id)
                )

    def handle(self, message: InboundMessage) -> None:
        with self._locks_guard:
            customer_lock = self._customer_locks[message.wa_id]
        with customer_lock:
            self._handle(message)

    def _handle(self, message: InboundMessage) -> None:
        now = self.clock()
        wa_id = message.wa_id
        if not self.repo.claim_message(message.id, now):
            logger.info("Duplicate message %s ignored", message.id)
            return
        logger.info("Message %s from %s (type=%s)", message.id, mask_phone(wa_id), message.type)
        if self.settings.mark_as_read:
            self._mark_read(message.id)

        text = message.text
        if (
            text is not None
            and wa_id == self.settings.owner_wa_id
            and text.lstrip().startswith("/")
        ):
            self._owner_command(text, now)
            return

        conversation = self.repo.get_conversation(wa_id)
        if conversation.handoff_until is not None:
            if now < conversation.handoff_until:
                logger.info("Handoff active for %s: the bot stays silent", mask_phone(wa_id))
                self._forward_to_owner(message, now)
                return
            self.repo.set_handoff(wa_id, None, now)  # the handoff expired
        self._route(message, conversation, now)

    def _forward_to_owner(self, message: InboundMessage, now: datetime) -> None:
        """While a person handles the chat, relay the customer's messages to the owner."""
        if self.rate_limiter.check(message.wa_id, now.timestamp()) is not RateDecision.ALLOW:
            return
        text = message.text if message.text is not None else f"[{message.type}]"
        self._notify_owner(
            message.wa_id,
            t(
                "owner_forward",
                self.business.owner_language,
                phone=message.wa_id,
                text=truncate_text(text, _OWNER_LAST_MESSAGE_CHARS * 3),
            ),
        )

    def _route(self, message: InboundMessage, conversation: Conversation, now: datetime) -> None:
        wa_id, text = message.wa_id, message.text
        lang: Lang = conversation.lang or self.business.default_language
        if text is not None:
            lang = resolve_language(text, conversation.lang, self.business.default_language)
            if lang != conversation.lang:
                self.repo.set_language(wa_id, lang, now)
            if is_handoff_request(text):
                self._start_handoff(message, text, lang, now)
                return

        decision = self.rate_limiter.check(wa_id, now.timestamp())
        if decision is RateDecision.NOTIFY:
            logger.info("Rate limit reached for %s", mask_phone(wa_id))
            self._send(wa_id, t("rate_limited", lang))
            return
        if decision is RateDecision.DROP:
            return

        if text is None:
            self._send(wa_id, t("unsupported_type", lang))
            return

        state = BookingState.from_json(conversation.flow_state)
        if state is not None:
            self._finish_step(wa_id, self.booking.handle(state, text, lang, now), lang, now)
            return
        if is_booking_intent(text):
            self._finish_step(wa_id, self.booking.start(lang), lang, now)
            return
        if is_greeting(text):
            self._send(wa_id, t("greeting", lang, business=self.business.name))
            return
        self._answer(wa_id, text, lang, now)

    # --- routes ---------------------------------------------------------------------------------
    def _answer(self, wa_id: str, text: str, lang: Lang, now: datetime) -> None:
        passages = self.knowledge.search(
            text,
            lang,
            k=self.settings.retrieval_top_k,
            fallback_below=self.settings.offline_min_score,
        )
        history = [
            Turn("user" if role == "user" else "assistant", body)
            for role, body in self.repo.recent_messages(wa_id, 2 * self.settings.history_turns)
        ]
        reply = self.llm.answer(AnswerRequest(text, lang, passages, history))
        self._send(wa_id, reply)
        self.repo.add_message(wa_id, "user", text, now)
        self.repo.add_message(wa_id, "assistant", reply, now)
        logger.info(
            "Answered %s with the %s provider (%d passages, best score %.2f)",
            mask_phone(wa_id),
            self.llm.name,
            len(passages),
            passages[0].score if passages else 0.0,
        )

    def _finish_step(self, wa_id: str, result: FlowResult, lang: Lang, now: datetime) -> None:
        reply = result.reply
        if result.confirmed is not None:
            request = result.confirmed
            booking_id = self.repo.add_booking(
                BookingRecord(
                    wa_id=wa_id,
                    service_id=request.service.id,
                    service_name=request.service.name(self.business.default_language),
                    date=request.day.isoformat(),
                    time=f"{request.at:%H:%M}",
                    patient_name=request.patient_name,
                    lang=lang,
                ),
                now,
            )
            reply = t("booking_saved", lang, id=booking_id)
            logger.info("Booking request #%d saved for %s", booking_id, mask_phone(wa_id))
            owner_lang = self.business.owner_language
            self._notify_owner(
                wa_id,
                t(
                    "owner_booking_alert",
                    owner_lang,
                    id=booking_id,
                    phone=wa_id,
                    service=request.service.name(owner_lang),
                    when=request.when(owner_lang),
                    name=request.patient_name,
                ),
            )
        state = result.state.to_json() if result.state is not None else None
        self.repo.set_flow_state(wa_id, state, now)
        self._send(wa_id, reply)

    def _start_handoff(self, message: InboundMessage, text: str, lang: Lang, now: datetime) -> None:
        wa_id = message.wa_id
        ttl_hours = self.settings.handoff_ttl_hours
        self.repo.set_flow_state(wa_id, None, now)
        self.repo.set_handoff(wa_id, now + timedelta(hours=ttl_hours), now)
        hours = self.business.hours
        after_hours = not hours.is_open(now)
        reply = t("handoff_started", lang)
        if after_hours:
            opening = hours.next_opening(now)
            if opening.date() == hours.local(now).date():
                reply += " " + t("handoff_after_hours_today", lang, time=f"{opening:%H:%M}")
            else:
                reply += " " + t("handoff_after_hours", lang, next_open=format_when(opening, lang))
        self._send(wa_id, reply)
        name = f" ({message.profile_name})" if message.profile_name else ""
        notified = self._notify_owner(
            wa_id,
            t(
                "owner_handoff_alert",
                self.business.owner_language,
                phone=wa_id,
                name=name,
                last=truncate_text(text, _OWNER_LAST_MESSAGE_CHARS),
                hours=f"{ttl_hours:g}",
            ),
        )
        handoff_id = self.repo.add_handoff(
            wa_id, lang, after_hours=after_hours, owner_notified=notified, now=now
        )
        logger.info(
            "Handoff #%d for %s (after_hours=%s, owner_notified=%s)",
            handoff_id,
            mask_phone(wa_id),
            after_hours,
            notified,
        )

    def _owner_command(self, text: str, now: datetime) -> None:
        owner = self.settings.owner_wa_id
        lang = self.business.owner_language
        parts = text.strip().split(maxsplit=2)
        command = parts[0].lower()
        if command == "/bot" and len(parts) == 3 and parts[1].lower() == "on":
            phone = _digits(parts[2])
            if phone:
                self.repo.set_handoff(phone, None, now)
                logger.info("Owner turned the bot back on for %s", mask_phone(phone))
                self._send(owner, t("owner_bot_on", lang, phone=phone))
                return
        if command == "/reply" and len(parts) == 3:
            phone = _digits(parts[1])
            if phone:
                delivered = self._send(phone, parts[2].strip())
                logger.info(
                    "Owner reply to %s through the bot (accepted=%s)", mask_phone(phone), delivered
                )
                key = "owner_reply_sent" if delivered else "owner_reply_failed"
                self._send(owner, t(key, lang, phone=phone))
                return
        self._send(owner, t("owner_usage", lang))

    # --- outbound -------------------------------------------------------------------------------
    def _notify_owner(self, customer: str, text: str) -> bool:
        owner = self.settings.owner_wa_id
        if not owner or owner == customer:
            return False
        return self._send(owner, text)

    def _send(self, to: str, body: str) -> bool:
        try:
            self.sender.send_text(to, truncate_text(body))
        except WhatsAppAPIError as exc:
            logger.error("Could not send a message to %s: %s", mask_phone(to), exc.message)
            return False
        return True

    def _mark_read(self, message_id: str) -> None:
        try:
            self.sender.mark_as_read(message_id)
        except WhatsAppAPIError as exc:
            logger.warning("Could not mark message %s as read: %s", message_id, exc.message)

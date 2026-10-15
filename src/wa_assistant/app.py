"""FastAPI application: the WhatsApp webhook, admin endpoints and a health check.

Run with ``uvicorn wa_assistant.app:create_app --factory``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from wa_assistant import __version__
from wa_assistant.config import Settings, load_business_config
from wa_assistant.engine import Clock, Engine, utc_now
from wa_assistant.knowledge import KnowledgeBase
from wa_assistant.llm import LLMProvider, build_provider
from wa_assistant.logging_utils import configure_logging
from wa_assistant.payloads import WebhookPayload, extract_messages
from wa_assistant.security import bearer_token, tokens_match, verify_signature
from wa_assistant.storage import Repository
from wa_assistant.whatsapp import MessageSender, WhatsAppClient

logger = logging.getLogger(__name__)


def _webhook_router(engine: Engine, settings: Settings) -> APIRouter:
    router = APIRouter()
    app_secret = settings.app_secret.get_secret_value()
    skip_signature = not app_secret and settings.env == "dev"
    if skip_signature:
        logger.warning(
            "WA_APP_SECRET is empty and WA_ENV=dev: webhook signatures are NOT verified. "
            "Never run like this in production."
        )
    elif not app_secret:
        logger.error("WA_APP_SECRET is empty: every webhook POST will be rejected with 401")
    if not settings.verify_token.get_secret_value():
        logger.warning("WA_VERIFY_TOKEN is empty: webhook verification (GET /webhook) will fail")

    @router.get("/webhook", response_class=PlainTextResponse)
    def verify_webhook(
        mode: Annotated[str | None, Query(alias="hub.mode")] = None,
        token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
        challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
    ) -> str:
        expected = settings.verify_token.get_secret_value()
        if mode == "subscribe" and challenge is not None and tokens_match(token, expected):
            logger.info("Webhook verified by Meta")
            return challenge
        raise HTTPException(status_code=403, detail="Forbidden")

    @router.post("/webhook")
    async def receive_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        signature: Annotated[str | None, Header(alias="X-Hub-Signature-256")] = None,
    ) -> dict[str, str]:
        body = await request.body()
        if not skip_signature and not verify_signature(body, signature, app_secret):
            logger.warning("Rejected a webhook POST with a missing or invalid signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
        try:
            raw = json.loads(body)
        except ValueError as exc:  # invalid JSON or invalid UTF-8
            raise HTTPException(status_code=400, detail="Body is not valid JSON") from exc
        try:
            payload = WebhookPayload.model_validate(raw)
        except ValidationError:
            # Signed by Meta but not a shape we handle: acknowledge so Meta does not retry it.
            logger.warning("Ignored a webhook payload with an unexpected shape")
            return {"status": "ignored"}
        messages, statuses = extract_messages(payload)
        if statuses:
            logger.debug("Ignored %d status update(s)", statuses)
        if messages:
            # Reply 200 right away; Meta retries webhooks that answer slowly.
            background_tasks.add_task(engine.process, messages)
        return {"status": "received"}

    return router


def _admin_router(repo: Repository, settings: Settings) -> APIRouter:
    def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = settings.admin_token.get_secret_value()
        if not expected:  # admin API disabled
            raise HTTPException(status_code=404, detail="Not Found")
        if not tokens_match(bearer_token(authorization), expected):
            raise HTTPException(
                status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"}
            )

    router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

    @router.get("/bookings")
    def list_bookings(limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[dict[str, Any]]:
        return repo.list_bookings(limit)

    @router.get("/handoffs")
    def list_handoffs(limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[dict[str, Any]]:
        return repo.list_handoffs(limit)

    return router


def create_app(
    settings: Settings | None = None,
    *,
    sender: MessageSender | None = None,
    llm: LLMProvider | None = None,
    clock: Clock = utc_now,
) -> FastAPI:
    """Build the application. Tests inject a fake ``sender``, ``llm`` and ``clock``."""
    settings = settings or Settings()
    configure_logging(settings.log_level)
    business = load_business_config(settings.data_dir / "business.toml")
    knowledge = KnowledgeBase.load(settings.data_dir / "knowledge")
    repo = Repository(settings.db_path)
    owned_client: WhatsAppClient | None = None
    if sender is None:
        owned_client = WhatsAppClient(
            access_token=settings.access_token.get_secret_value(),
            phone_number_id=settings.phone_number_id,
            graph_version=settings.graph_version,
            base_url=settings.graph_base_url,
            timeout=settings.http_timeout_seconds,
        )
        sender = owned_client
    llm = llm or build_provider(settings, business.name)
    engine = Engine(
        settings=settings,
        business=business,
        repo=repo,
        knowledge=knowledge,
        llm=llm,
        sender=sender,
        clock=clock,
    )
    logger.info(
        "Started %s (env=%s, answers=%s, %d knowledge passages)",
        business.name,
        settings.env,
        llm.name,
        len(knowledge.passages),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        repo.close()
        if owned_client is not None:
            owned_client.close()

    is_dev = settings.env == "dev"
    app = FastAPI(
        title="WhatsApp AI Assistant",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if is_dev else None,
        redoc_url=None,
        openapi_url="/openapi.json" if is_dev else None,
    )
    app.state.engine = engine

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(_webhook_router(engine, settings))
    app.include_router(_admin_router(repo, settings))
    return app

from __future__ import annotations

import json
from collections.abc import Callable

import httpx2
import pytest

from wa_assistant.whatsapp import (
    MAX_TEXT_LENGTH,
    RecordingSender,
    WhatsAppAPIError,
    WhatsAppClient,
    truncate_text,
)

Handler = Callable[[httpx2.Request], httpx2.Response]


def make_client(handler: Handler) -> WhatsAppClient:
    return WhatsAppClient(
        access_token="test-token",
        phone_number_id="100000000000001",
        graph_version="v23.0",
        retry_backoff=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


class Recorder:
    def __init__(self, *responses: httpx2.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def ok() -> httpx2.Response:
    return httpx2.Response(200, json={"messages": [{"id": "wamid.out1"}]})


def test_truncate_text() -> None:
    assert truncate_text("hola") == "hola"
    exact = "x" * MAX_TEXT_LENGTH
    assert truncate_text(exact) == exact
    long = truncate_text("y" * 5000)
    assert len(long) == MAX_TEXT_LENGTH
    assert long.endswith("…")


def test_send_text_payload_url_and_auth() -> None:
    recorder = Recorder(ok())
    client = make_client(recorder)
    client.send_text("15550100001", "Hola")
    request = recorder.requests[0]
    assert str(request.url) == "https://graph.facebook.com/v23.0/100000000000001/messages"
    assert request.headers["Authorization"] == "Bearer test-token"
    assert json.loads(request.content) == {
        "messaging_product": "whatsapp",
        "to": "15550100001",
        "type": "text",
        "text": {"body": "Hola", "preview_url": False},
    }
    client.close()


def test_send_text_truncates_to_the_whatsapp_limit() -> None:
    recorder = Recorder(ok())
    make_client(recorder).send_text("15550100001", "z" * 10_000)
    body = json.loads(recorder.requests[0].content)["text"]["body"]
    assert len(body) == MAX_TEXT_LENGTH


def test_mark_as_read_payload() -> None:
    recorder = Recorder(ok())
    make_client(recorder).mark_as_read("wamid.in1")
    assert json.loads(recorder.requests[0].content) == {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": "wamid.in1",
    }


def test_retries_once_on_5xx() -> None:
    recorder = Recorder(httpx2.Response(503), ok())
    make_client(recorder).send_text("15550100001", "Hola")
    assert len(recorder.requests) == 2


def test_gives_up_after_two_5xx() -> None:
    recorder = Recorder(httpx2.Response(500, json={"error": {"message": "Service unavailable"}}))
    with pytest.raises(WhatsAppAPIError) as info:
        make_client(recorder).send_text("15550100001", "Hola")
    assert info.value.status_code == 500
    assert len(recorder.requests) == 2


def test_retries_once_on_connection_errors() -> None:
    recorder = Recorder(httpx2.ConnectError("refused"), ok())
    make_client(recorder).send_text("15550100001", "Hola")
    assert len(recorder.requests) == 2


def test_raises_after_two_connection_errors() -> None:
    recorder = Recorder(httpx2.ReadTimeout("slow"))
    with pytest.raises(WhatsAppAPIError) as info:
        make_client(recorder).send_text("15550100001", "Hola")
    assert info.value.status_code is None
    assert len(recorder.requests) == 2


def test_4xx_raises_with_the_graph_error_message_and_does_not_retry() -> None:
    error = {
        "error": {
            "message": "(#131030) Recipient phone number not in allowed list",
            "type": "OAuthException",
            "code": 131030,
        }
    }
    recorder = Recorder(httpx2.Response(400, json=error))
    with pytest.raises(WhatsAppAPIError, match="not in allowed list") as info:
        make_client(recorder).send_text("15550100001", "Hola")
    assert info.value.status_code == 400
    assert len(recorder.requests) == 1


def test_4xx_with_a_non_json_body() -> None:
    recorder = Recorder(httpx2.Response(401, text="Unauthorized"))
    with pytest.raises(WhatsAppAPIError, match="Unauthorized"):
        make_client(recorder).mark_as_read("wamid.1")


def test_4xx_with_unexpected_json() -> None:
    recorder = Recorder(httpx2.Response(404, json=["nope"]))
    with pytest.raises(WhatsAppAPIError, match="nope"):
        make_client(recorder).mark_as_read("wamid.1")


def test_recording_sender() -> None:
    sender = RecordingSender()
    sender.send_text("1", "a")
    sender.send_text("2", "b")
    sender.mark_as_read("wamid.1")
    assert sender.texts_to("1") == ["a"]
    assert sender.read == ["wamid.1"]


def test_default_client_builds_its_own_http_client() -> None:
    client = WhatsAppClient(access_token="t", phone_number_id="1", graph_version="v99.0")
    assert client.url == "https://graph.facebook.com/v99.0/1/messages"
    client.close()

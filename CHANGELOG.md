# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-15

First public release.

### Added
- Grounded answers from the business's own Markdown documents (Spanish,
  Portuguese and English), ranked with a pure-Python Okapi BM25. Every answer
  ends with a source tag, and when nothing relevant is found the bot says so
  and offers a person instead of guessing.
- Two answer providers behind one interface: Claude through the official
  `anthropic` SDK, falling back to the offline answer on any API error, and a
  deterministic offline provider used by the tests, the simulator and servers
  without an API key.
- Language detection (es / pt / en) that stays fixed for a conversation unless
  a message clearly switches language.
- Appointment requests through a deterministic state machine (service, date,
  time, name, confirmation), checked against opening hours and a 60-day window.
- Human handoff: the owner gets a WhatsApp alert, the bot stays silent for that
  customer and relays their messages, and the owner can reply through the bot.
- WhatsApp Cloud API webhook with `X-Hub-Signature-256` verification, an
  immediate 200 with background processing, idempotency over message ids,
  per-customer ordering and a per-customer rate limit.
- Admin endpoints for bookings and handoffs behind a bearer token compared in
  constant time, `GET /healthz`, and `/docs` only in development.
- A terminal simulator of the whole engine (`python -m wa_assistant.simulate`)
  that needs no keys.

### Security
- Customer text is treated as untrusted, escaped and wrapped before it reaches
  the model; the conversation history sent to the model is bounded.
- Logs mask phone numbers and never contain message bodies.
- CI runs secret, dependency and container-image scans, with Dependabot keeping
  dependencies current.

[1.0.0]: https://github.com/mojtaba-py-code/whatsapp-ai-assistant/releases/tag/v1.0.0

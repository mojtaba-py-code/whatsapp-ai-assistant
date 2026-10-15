"""Heuristic language detection for Spanish, Portuguese and English.

Two kinds of signals are counted:

* **marker words** - frequent words that belong to one language only (shared words such as
  "de", "que", "para" or "como" are left out on purpose), and
* **character signals** - "ñ", "¿" and "¡" are Spanish; "ã", "õ" and "ç" are Portuguese,
  as are the digraphs "nh" and "lh".

The language is *sticky* per conversation: once known, it only changes when a message
carries clear evidence of another language and none of the current one.
"""

from __future__ import annotations

from wa_assistant.text import LANGUAGES, Lang, words


def _set(raw: str) -> frozenset[str]:
    return frozenset(raw.split())


_MARKERS: dict[Lang, frozenset[str]] = {
    "es": _set(
        """
        el los las del y con una un es estan estoy hola gracias quiero necesito cuanto cuesta
        donde tienen tiene hay mi yo muy pero puedo hacen atienden abren cuando manana si usted
        ustedes vos podes buenas buenos turno cita precio precios ninos nino hoy semana
        sabado domingo ayuda dolor muela diente
        """
    ),
    "pt": _set(
        """
        voce voces vc vcs nao obrigado obrigada ola oi quero preciso quanto custa onde fica tem
        sim amanha bom boa noite um uma com meu minha posso pode e ao em isso esse essa muito
        eu gostaria tudo bem hoje criancas crianca dente dor atendente preco precos
        agendamento sabado domingo
        """
    ),
    "en": _set(
        """
        the is are what how much where does you your i my hello hi thanks thank please can
        could book appointment open when which want need have price prices cost tomorrow today
        yes to for and of in with it this that there we kids children tooth pain human
        """
    ),
}
# Words that appear in more than one marker list carry no signal.
_SHARED = {
    word
    for word in set().union(*_MARKERS.values())
    if sum(word in m for m in _MARKERS.values()) > 1
}

_CHAR_SIGNALS: dict[Lang, tuple[str, ...]] = {
    "es": ("ñ", "¿", "¡"),
    "pt": ("ã", "õ", "ç"),
    "en": (),
}


def language_scores(text: str) -> dict[Lang, int]:
    """Evidence count per language for ``text``."""
    scores: dict[Lang, int] = dict.fromkeys(LANGUAGES, 0)
    tokens = words(text)
    for token in tokens:
        if token in _SHARED:
            continue
        for lang in LANGUAGES:
            if token in _MARKERS[lang]:
                scores[lang] += 1
        if len(token) > 3 and ("nh" in token or "lh" in token):
            scores["pt"] += 1
    lowered = text.casefold()
    for lang in LANGUAGES:
        scores[lang] += 2 * sum(lowered.count(ch) for ch in _CHAR_SIGNALS[lang])
    return scores


def detect_language(text: str) -> Lang | None:
    """The language with the strictly highest evidence, or ``None`` when there is no clear winner."""
    scores = language_scores(text)
    ranked = sorted(LANGUAGES, key=lambda lang: scores[lang], reverse=True)
    best, runner_up = ranked[0], ranked[1]
    if scores[best] == 0 or scores[best] == scores[runner_up]:
        return None
    return best


def resolve_language(text: str, current: Lang | None, default: Lang) -> Lang:
    """Pick the conversation language after a new message (sticky once known)."""
    detected = detect_language(text)
    if current is None:
        return detected or default
    if detected is None or detected == current:
        return current
    scores = language_scores(text)
    if scores[detected] >= 2 and scores[current] == 0:
        return detected
    return current


__all__ = ["detect_language", "language_scores", "resolve_language"]

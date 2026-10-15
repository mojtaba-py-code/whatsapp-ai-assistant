from __future__ import annotations

import pytest

from wa_assistant.language import detect_language, language_scores, resolve_language
from wa_assistant.text import Lang, fold, stem, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hola, ¿cuánto cuesta una limpieza?", "es"),
        ("¿Dónde están ubicados?", "es"),
        ("Necesito un turno para mañana", "es"),
        ("quiero saber los precios", "es"),
        ("Olá, quanto custa um clareamento?", "pt"),
        ("Vocês atendem crianças?", "pt"),
        ("Preciso marcar uma consulta amanhã", "pt"),
        ("onde fica a clínica?", "pt"),
        ("Hello, how much is a cleaning?", "en"),
        ("Do you accept insurance?", "en"),
        ("I need an appointment tomorrow", "en"),
        ("Where are you located?", "en"),
    ],
)
def test_detects_spanish_portuguese_and_english(text: str, expected: Lang) -> None:
    assert detect_language(text) == expected


@pytest.mark.parametrize("text", ["ok", "10:30", "1", "👍", ""])
def test_no_clear_signal_returns_none(text: str) -> None:
    assert detect_language(text) is None


def test_character_signals_count() -> None:
    assert language_scores("ñ")["es"] == 2
    assert language_scores("ç")["pt"] == 2
    assert language_scores("¿?")["es"] == 2


def test_portuguese_digraphs_are_a_signal() -> None:
    assert language_scores("trabalho")["pt"] == 1


def test_resolve_uses_default_until_a_language_is_detected() -> None:
    assert resolve_language("1", None, "es") == "es"
    assert resolve_language("Olá, tudo bem?", None, "es") == "pt"


def test_resolve_is_sticky_for_short_or_ambiguous_messages() -> None:
    assert resolve_language("ok", "pt", "es") == "pt"
    assert resolve_language("sim", "es", "es") == "es"  # one weak word does not switch
    assert resolve_language("Ana García", "en", "es") == "en"


def test_resolve_switches_on_clear_evidence() -> None:
    assert resolve_language("Do you open on Sunday? I need help", "es", "es") == "en"
    assert resolve_language("¿Hay turnos para mañana?", "pt", "es") == "es"


def test_fold_and_tokenize() -> None:
    assert fold("Mañana ATENÇÃO Niños") == "manana atencao ninos"
    assert tokenize("¿Atienden a los niños?", "es") == ["atienden", "nino"]
    assert stem("sociales") == "social"
    assert stem("cartoes") == "cartao"
    assert stem("prices") == "price"
    assert stem("class") == "class"
    assert stem("mes") == "mes"

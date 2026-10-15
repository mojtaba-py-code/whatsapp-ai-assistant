from __future__ import annotations

from pathlib import Path

import pytest

from wa_assistant.knowledge import BM25Index, KnowledgeBase, Passage, chunk_markdown
from wa_assistant.text import Lang, tokenize


def test_every_language_has_the_same_documents(knowledge: KnowledgeBase) -> None:
    docs: dict[str, set[str]] = {}
    for passage in knowledge.passages:
        docs.setdefault(passage.lang, set()).add(passage.doc_id.split("/")[1])
    assert set(docs) == {"es", "pt", "en"}
    assert docs["es"] == docs["pt"] == docs["en"]
    assert len(docs["es"]) == 8


def test_chunking_by_heading() -> None:
    text = "# Horarios\n\nIntro.\n\n## Semana\n\nLunes a viernes.\n\n## Fin de semana\n\nSábados.\n"
    passages = chunk_markdown(text, "es/hours", "es")
    assert [(p.title, p.heading, p.body) for p in passages] == [
        ("Horarios", "Horarios", "Intro."),
        ("Horarios", "Semana", "Lunes a viernes."),
        ("Horarios", "Fin de semana", "Sábados."),
    ]


def test_document_without_title_uses_the_file_name() -> None:
    passages = chunk_markdown("Just text.", "en/notes", "en")
    assert passages[0].title == "notes"


@pytest.mark.parametrize(
    ("lang", "question", "doc_id"),
    [
        ("es", "¿Cuánto cuesta una limpieza?", "es/prices"),
        ("es", "¿A qué hora abren el sábado?", "es/hours"),
        ("es", "¿Dónde están?", "es/location"),
        ("es", "¿Aceptan obra social?", "es/payments"),
        ("es", "¿Aceptan tarjeta?", "es/payments"),
        ("es", "¿Cómo cancelo un turno?", "es/appointments"),
        ("es", "¿Atienden urgencias?", "es/emergencies"),
        ("es", "¿Desde qué edad atienden niños?", "es/children"),
        ("es", "¿Qué tengo que traer a la primera consulta?", "es/first-visit"),
        ("pt", "Quanto custa um clareamento?", "pt/prices"),
        ("pt", "Vocês abrem no sábado?", "pt/hours"),
        ("pt", "Onde fica a clínica?", "pt/location"),
        ("pt", "Vocês aceitam cartão?", "pt/payments"),
        ("pt", "Atendem crianças?", "pt/children"),
        ("en", "How much is a cleaning?", "en/prices"),
        ("en", "Are you open on Saturday?", "en/hours"),
        ("en", "Where are you located?", "en/location"),
        ("en", "Do you accept insurance?", "en/payments"),
        ("en", "What should I bring to my first visit?", "en/first-visit"),
        ("en", "How do I cancel an appointment?", "en/appointments"),
    ],
)
def test_bm25_ranks_the_right_document_first(
    knowledge: KnowledgeBase, lang: Lang, question: str, doc_id: str
) -> None:
    results = knowledge.search(question, lang)
    assert results, question
    assert results[0].passage.doc_id == doc_id
    assert results[0].passage.lang == lang


def test_scores_are_sorted_and_bounded_by_k(knowledge: KnowledgeBase) -> None:
    results = knowledge.search("¿Atienden urgencias de niños los sábados?", "es", k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score > 0


def test_falls_back_to_all_languages(knowledge: KnowledgeBase) -> None:
    results = knowledge.search("emergency", "es")
    assert results[0].passage.doc_id == "en/emergencies"


def test_unrelated_questions_retrieve_nothing(knowledge: KnowledgeBase) -> None:
    assert knowledge.search("¿Cuál es la capital de Francia?", "es") == []
    assert knowledge.search("Do you sell bicycles?", "en") == []


def test_bm25_index_basics() -> None:
    a = Passage("en/a", "en", "A", "A", "apples and pears")
    b = Passage("en/b", "en", "B", "B", "pears pears pears")
    index = BM25Index([(a, tokenize(a.body, "en")), (b, tokenize(b.body, "en"))])
    assert len(index) == 2
    assert [r.passage.doc_id for r in index.search(tokenize("apples", "en"), 5)] == ["en/a"]
    assert index.search(tokenize("banana", "en"), 5) == []
    assert index.search(tokenize("pear", "en"), 5)[0].passage.doc_id == "en/b"


def test_load_requires_documents(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no knowledge documents"):
        KnowledgeBase.load(tmp_path)

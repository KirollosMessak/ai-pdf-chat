import pytest

from rag import Chunk, Retriever, format_context, split_chunks


def test_split_keeps_page_numbers_and_overlaps():
    pages = [Chunk("a.pdf", 1, "word " * 600), Chunk("a.pdf", 2, "short page")]
    chunks = split_chunks(pages, size=1000, overlap=200)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 1000 for c in chunks)
    assert len([c for c in chunks if c.page == 1]) == 4


def test_search_ranks_relevant_chunk_first():
    chunks = [
        Chunk("policy.pdf", 1, "Employees get 21 days of paid annual leave."),
        Chunk("policy.pdf", 2, "The office opens at 9am and closes at 5pm."),
        Chunk("policy.pdf", 3, "Expense reports are due by the 5th of each month."),
    ]
    top = Retriever(chunks).search("how many days of annual leave", k=1)
    assert top[0].page == 1


def test_empty_documents_raise_clear_error():
    with pytest.raises(ValueError, match="No text found"):
        Retriever([])


def test_format_context_includes_citation_metadata():
    out = format_context([Chunk("x.pdf", 7, "hello")])
    assert 'file="x.pdf"' in out and 'page="7"' in out


def test_search_works_with_only_two_chunks():
    chunks = [
        Chunk("faq.pdf", 1, "Refund policy: customers may return items within 30 days."),
        Chunk("faq.pdf", 2, "Shipping takes 3 to 5 business days within Egypt."),
    ]
    assert Retriever(chunks).search("how long is shipping", k=1)[0].page == 2

"""Offline tests for slop.fetch_article.

These tests exercise only the local helpers (URL validation, paragraph
splitting, word counting) and never call the TinyFish API.

Run with:
    python -m tests.test_fetch_article
or:
    pytest tests/test_fetch_article.py
"""

from slop.fetch_article import (
    FetchError,
    cleanup_paragraphs,
    fetch_article_text,
    split_paragraphs,
    validate_url,
    word_count,
)


def test_validate_url_accepts_http_and_https():
    validate_url("http://example.com")
    validate_url("https://example.com/path?query=1")


def test_validate_url_rejects_invalid():
    for bad in ["ftp://example.com", "example.com", "", "http:/example", "mailto:x"]:
        try:
            validate_url(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_split_paragraphs_blank_lines():
    text = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
    assert split_paragraphs(text) == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]


def test_split_paragraphs_removes_markdown():
    text = "# Heading\n\n**bold** and *italic* text.\n\n## Sub-heading"
    assert split_paragraphs(text) == [
        "Heading",
        "bold and italic text.",
        "Sub-heading",
    ]


def test_split_paragraphs_skips_empty():
    assert split_paragraphs("\n\n   \n\n") == []


def test_word_count():
    assert word_count(["hello world", "foo bar baz"]) == 5
    assert word_count([]) == 0
    assert word_count(["single"]) == 1


def test_cleanup_paragraphs_removes_furniture():
    text = (
        "This is a real article paragraph with enough words to pass the filter.\n\n"
        "Albert Einstein\n"
        "Marie Curie\n"
        "Isaac Newton\n"
        "Charles Darwin\n"
        "Nikola Tesla\n"
        "Stephen Hawking\n\n"
        "| col1 | col2 |\n|---|---|\n| a | b |\n\n"
        "Retrieved from https://example.com on 2024-01-01\n\n"
        "Categories: Topic, Stuff\n\n"
        "short\n\n"
        "[link text](https://example.com) and **bold** text with several more words here to pass the filter.\n\n"
        "## References\n\n"
        "Some reference that should be dropped.\n\n"
    )
    assert cleanup_paragraphs(text) == [
        "This is a real article paragraph with enough words to pass the filter.",
        "link text and bold text with several more words here to pass the filter.",
    ]


def test_cleanup_paragraphs_stops_at_list_headings():
    text = (
        "This is a normal sentence with more than twelve words here today.\n\n"
        "## Software\n\n"
        "Some software list that should be dropped.\n\n"
    )
    assert cleanup_paragraphs(text) == [
        "This is a normal sentence with more than twelve words here today.",
    ]


ARTICLE_URL = "https://en.wikipedia.org/wiki/Example"


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _selector_not_matched(url):
    return {
        "errors": [{"url": url, "error": "selector_not_matched", "status": 200}],
        "results": [],
    }


def _success(url, text):
    return {
        "errors": [],
        "results": [{"url": url, "title": "Example", "text": text}],
    }


def _long_article_text():
    return (
        "In 1969, the Apollo 11 mission landed the first humans on the Moon, "
        "marking a historic achievement in space exploration.\n\n"
        "The crew included Neil Armstrong, Buzz Aldrin, and Michael Collins, "
        "who traveled aboard the Saturn V rocket from Kennedy Space Center.\n\n"
        "Armstrong stepped onto the lunar surface and declared it a giant "
        "leap for mankind, while millions watched the broadcast on television "
        "around the world.\n"
    )


def _make_fake_post(responses):
    calls = []

    def fake_post(endpoint, headers, json, timeout):
        calls.append(json)
        return _FakeResponse(200, responses[len(calls) - 1])

    return fake_post, calls


def test_fetch_article_fallback_without_selectors():
    import slop.fetch_article as fa

    responses = [_selector_not_matched(ARTICLE_URL), _success(ARTICLE_URL, _long_article_text())]
    fake_post, calls = _make_fake_post(responses)
    original_post = fa.requests.post
    original_api_key = fa._api_key
    fa.requests.post = fake_post
    fa._api_key = lambda: "test-key"
    try:
        result = fetch_article_text(ARTICLE_URL)
    finally:
        fa.requests.post = original_post
        fa._api_key = original_api_key

    assert len(calls) == 2
    assert "include_selectors" in calls[0]
    assert "include_selectors" not in calls[1]
    assert "exclude_selectors" in calls[1]
    assert result["word_count"] >= 50


def test_fetch_article_fallback_too_short():
    import slop.fetch_article as fa

    short_text = "Too few words here.\n"
    responses = [_selector_not_matched(ARTICLE_URL), _success(ARTICLE_URL, short_text)]
    fake_post, _calls = _make_fake_post(responses)
    original_post = fa.requests.post
    original_api_key = fa._api_key
    fa.requests.post = fake_post
    fa._api_key = lambda: "test-key"
    try:
        try:
            fetch_article_text(ARTICLE_URL)
        except FetchError as exc:
            assert "does not look like an article" in str(exc)
            assert "direct link" in str(exc)
            return
        raise AssertionError("expected FetchError for short fallback result")
    finally:
        fa.requests.post = original_post
        fa._api_key = original_api_key


def main():
    test_validate_url_accepts_http_and_https()
    test_validate_url_rejects_invalid()
    test_split_paragraphs_blank_lines()
    test_split_paragraphs_removes_markdown()
    test_split_paragraphs_skips_empty()
    test_word_count()
    test_cleanup_paragraphs_removes_furniture()
    test_cleanup_paragraphs_stops_at_list_headings()
    test_fetch_article_fallback_without_selectors()
    test_fetch_article_fallback_too_short()
    print("all offline tests passed")


if __name__ == "__main__":
    main()

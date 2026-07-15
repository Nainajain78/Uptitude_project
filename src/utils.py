"""Shared text utilities used across pipeline stages (extraction, summarization)."""


def split_paragraphs(text: str) -> list[str]:
    """Split normalized contract text into non-empty paragraphs on blank lines."""
    return [p for p in text.split("\n\n") if p.strip()]


def truncate_to_word_budget(text: str, max_words: int, head_ratio: float = 0.7) -> str:
    """Sensibly truncate long text to a word budget by keeping the head and tail.

    Keeps the first `head_ratio` fraction of the budget from the start of the
    document (where purpose/parties/recitals typically live) and the rest from
    the end (where termination/penalty/signature sections typically live),
    dropping the middle. This is a cheap alternative to full map-reduce
    summarization and works well for CUAD-style contracts, which front- and
    back-load the sections a summary needs to cover.

    Args:
        text: Full text to truncate.
        max_words: Target word budget for the returned text.
        head_ratio: Fraction of max_words to take from the start; the
            remainder is taken from the end.

    Returns:
        The truncated text, or the original text unchanged if it's already
        within the word budget.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    head_words = int(max_words * head_ratio)
    tail_words = max_words - head_words

    head = " ".join(words[:head_words])
    tail = " ".join(words[-tail_words:]) if tail_words > 0 else ""
    omitted = len(words) - head_words - tail_words

    return f"{head}\n\n[... {omitted} words omitted ...]\n\n{tail}"

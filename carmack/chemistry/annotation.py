"""Helpers for read-annotation headers and 0-based half-open position spans.

This module defines the small, dependency-free vocabulary used to annotate
reads with component positions: the position-key suffix convention and the
textual encoding of spans. It imports only from the standard library and must
not depend on the :mod:`carmack.io` package.
"""

POS_SUFFIX = "_POS"


def position_key(name: str) -> str:
    """Return the annotation header key for a component's position.

    Args:
        name: Identifier of the read component (e.g. ``"BC1"``).

    Returns:
        The component name with the canonical position suffix appended.
    """
    return f"{name}{POS_SUFFIX}"


def validate_span(start: int, end: int) -> None:
    """Validate that ``start`` and ``end`` form a legal half-open span.

    Args:
        start: Inclusive 0-based start position of the span.
        end: Exclusive 0-based end position of the span.

    Raises:
        ValueError: If ``start`` is negative or ``end`` is before ``start``.
    """
    if start < 0:
        raise ValueError(f"Span start must be non-negative, got {start}.")
    if end < start:
        raise ValueError(f"Span end must not precede start, got start={start}, end={end}.")


def format_span(start: int, end: int) -> str:
    """Render a 0-based half-open span as ``start:end`` text.

    Args:
        start: Inclusive 0-based start position of the span.
        end: Exclusive 0-based end position of the span. A zero-length span
            where ``end`` equals ``start`` is valid.

    Returns:
        The span encoded as ``"start:end"``.

    Raises:
        ValueError: If ``start`` is negative or ``end`` is before ``start``.
    """
    validate_span(start, end)
    return f"{start}:{end}"


def parse_span(text: str) -> tuple[int, int]:
    """Parse ``start:end`` span text into its integer bounds.

    This is the exact inverse of :func:`format_span`.

    Args:
        text: Span text of the form ``"start:end"``.

    Returns:
        A ``(start, end)`` tuple of the parsed 0-based half-open bounds.

    Raises:
        ValueError: If ``text`` does not contain exactly one colon, if either
            side is not an integer, if ``start`` is negative, or if ``end`` is
            before ``start``.
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Span text must be of the form 'start:end', got {text!r}.")
    start = int(parts[0])
    end = int(parts[1])
    validate_span(start, end)
    return (start, end)

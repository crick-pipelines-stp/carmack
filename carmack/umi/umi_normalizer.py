"""UMI length normalisation (SI-4 of the UMI epic).

umi_tools' clustering requires every UMI in one call to be the same length, but
extraction yields raw UMIs across the tolerance window ``[x - tol, x + tol]``.
This module forces each raw UMI onto the single canonical length ``x`` by
right-padding the short ones and right-trimming the long ones with a single
swappable sentinel, rejecting anything outside the window.

The sentinel :data:`UMI_PAD_CHAR` is the one place the padding character is
defined; changing it must be a one-line edit here.
"""

UMI_PAD_CHAR = "N"


def normalize_umi(raw: str, x: int, tol: int) -> str | None:
    """Force a raw UMI onto the canonical length ``x``.

    Args:
        raw: The raw UMI sequence as extracted.
        x: The canonical UMI length every normalised UMI is forced to.
        tol: The length tolerance; raw UMIs whose length lies outside
            ``[x - tol, x + tol]`` are rejected.

    Returns:
        A length-``x`` string: ``raw`` right-padded with :data:`UMI_PAD_CHAR`
        when shorter than ``x``, right-trimmed when longer, or unchanged when it
        already has length ``x``. Returns ``None`` when ``raw`` falls outside the
        length window.
    """
    length = len(raw)
    if length < x - tol or length > x + tol:
        return None
    if length < x:
        return raw + UMI_PAD_CHAR * (x - length)
    if length > x:
        return raw[:x]
    return raw

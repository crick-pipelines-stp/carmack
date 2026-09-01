"""Read-annotation value object for FASTQ headers (ST1 of issue #14)."""

from dataclasses import dataclass, field


@dataclass
class ReadAnnotation:
    """A parsed FASTQ header split into read id, tags, and passthrough tokens.

    A FASTQ header is a whitespace-separated sequence of tokens. The first
    token is the read id. Each remaining token is either a ``KEY=VALUE`` tag
    (when it contains ``=``) or a verbatim passthrough token (when it does not).

    Attributes:
        read_id: The first whitespace token of the header, or ``""`` when the
            header is empty or whitespace-only.
        tags: Key/value tags in insertion order; duplicate keys collapse to the
            last value seen.
        passthrough: Tokens without ``=``, preserved verbatim in input order.
    """

    read_id: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    passthrough: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, header: str) -> "ReadAnnotation":
        """Parse a FASTQ header into a read-annotation value object.

        The header is split on runs of whitespace. The first token becomes the
        read id (or ``""`` when there are no tokens). Each remaining token that
        contains ``=`` is stored as a tag, splitting on the first ``=`` only so
        that a value may itself contain ``=``; every other token is stored as a
        passthrough token.

        Args:
            header: The raw FASTQ header line to parse.

        Returns:
            A ``ReadAnnotation`` populated from the header.
        """
        tokens = header.split()
        read_id = tokens[0] if tokens else ""
        tags: dict[str, str] = {}
        passthrough: list[str] = []
        for token in tokens[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                tags[key] = value
            else:
                passthrough.append(token)
        return cls(read_id=read_id, tags=tags, passthrough=passthrough)

    def render(self) -> str:
        """Render the annotation back into a FASTQ header string.

        Tokens are emitted as read id, then passthrough tokens in order, then
        tag tokens formatted as ``KEY=VALUE`` in insertion order. An empty
        annotation renders to ``""``.

        Returns:
            The header string represented by this annotation.
        """
        tag_tokens = [f"{key}={value}" for key, value in self.tags.items()]
        parts = [self.read_id, *self.passthrough, *tag_tokens]
        return " ".join(part for part in parts if part != "")

    def get(self, key: str) -> str | None:
        """Return the value stored for a tag key.

        Args:
            key: The tag key to look up.

        Returns:
            The tag value, or ``None`` when the key is not present.
        """
        return self.tags.get(key)

    def set(self, key: str, value: str) -> None:
        """Add or overwrite a tag in place.

        Args:
            key: The tag key to set.
            value: The tag value to store.
        """
        self.tags[key] = value

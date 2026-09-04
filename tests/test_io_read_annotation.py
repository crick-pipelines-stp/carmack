"""Unit tests for the read-annotation value object."""

import ast
import unittest
from pathlib import Path

from carmack.io.read_annotation import ReadAnnotation

ILLUMINA_HEADER = "NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"


def collect_imported_module_names(source: str) -> list[str]:
    """Return the fully-qualified module names referenced by import statements.

    Args:
        source: Python source code to inspect.

    Returns:
        Every module name that the source imports, expanded so that
        ``from a.b import c`` yields both ``a.b`` and ``a.b.c``.
    """
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names.append(module)
            for alias in node.names:
                names.append(f"{module}.{alias.name}" if module else alias.name)
    return names


class TestReadAnnotationParse(unittest.TestCase):
    """Parsing of FASTQ headers into read-annotation value objects."""

    def test_parse_tag_token_populates_tags(self) -> None:
        """Tokens containing ``=`` are stored as key/value tags."""
        annotation = ReadAnnotation.parse("read1 BC=42 UMI=ACGT")
        self.assertEqual(annotation.read_id, "read1")
        self.assertEqual(annotation.tags, {"BC": "42", "UMI": "ACGT"})
        self.assertEqual(annotation.passthrough, [])

    def test_parse_passthrough_token_populates_passthrough(self) -> None:
        """Tokens without ``=`` are preserved verbatim as passthrough."""
        annotation = ReadAnnotation.parse("read1 1:N:0:ACGT")
        self.assertEqual(annotation.read_id, "read1")
        self.assertEqual(annotation.passthrough, ["1:N:0:ACGT"])
        self.assertEqual(annotation.tags, {})

    def test_parse_mixed_tokens_separates_tags_and_passthrough(self) -> None:
        """Mixed headers split tags and passthrough tokens correctly."""
        annotation = ReadAnnotation.parse("read1 1:N:0:ACGT BC1=42:50 UMI=ACGTAC")
        self.assertEqual(annotation.read_id, "read1")
        self.assertEqual(annotation.passthrough, ["1:N:0:ACGT"])
        self.assertEqual(annotation.tags, {"BC1": "42:50", "UMI": "ACGTAC"})

    def test_parse_illumina_comment_lands_in_passthrough(self) -> None:
        """The Illumina comment token is preserved as passthrough, not a tag."""
        annotation = ReadAnnotation.parse(ILLUMINA_HEADER)
        self.assertEqual(annotation.read_id, "NB501505:171:H3KMGAFX3:1:21208:17616:17963")
        self.assertEqual(annotation.passthrough, ["1:N:0:AGATCTCGGT"])
        self.assertEqual(annotation.tags, {})
        self.assertEqual(annotation.render(), ILLUMINA_HEADER)

    def test_parse_empty_header_yields_empty_fields(self) -> None:
        """An empty header parses to empty fields and round-trips."""
        annotation = ReadAnnotation.parse("")
        self.assertEqual(annotation.read_id, "")
        self.assertEqual(annotation.tags, {})
        self.assertEqual(annotation.passthrough, [])
        self.assertEqual(ReadAnnotation.parse(annotation.render()), annotation)

    def test_parse_whitespace_only_header_yields_empty_fields(self) -> None:
        """A whitespace-only header parses to empty fields and round-trips."""
        annotation = ReadAnnotation.parse("   ")
        self.assertEqual(annotation.read_id, "")
        self.assertEqual(annotation.tags, {})
        self.assertEqual(annotation.passthrough, [])
        self.assertEqual(ReadAnnotation.parse(annotation.render()), annotation)

    def test_parse_no_tag_header_yields_only_read_id(self) -> None:
        """A single-token header yields only a read id."""
        annotation = ReadAnnotation.parse("read1")
        self.assertEqual(annotation.read_id, "read1")
        self.assertEqual(annotation.tags, {})
        self.assertEqual(annotation.passthrough, [])

    def test_parse_duplicate_keys_last_value_wins(self) -> None:
        """When a key repeats, the last value is retained."""
        annotation = ReadAnnotation.parse("read1 K=1 K=2")
        self.assertEqual(annotation.tags, {"K": "2"})

    def test_parse_value_with_equals_splits_on_first_equals(self) -> None:
        """A tag value may itself contain ``=`` because only the first splits."""
        annotation = ReadAnnotation.parse("read1 UMI=AC=GT")
        self.assertEqual(annotation.tags, {"UMI": "AC=GT"})
        self.assertIsNone(annotation.get("AC"))


class TestReadAnnotationRender(unittest.TestCase):
    """Rendering of read-annotation value objects back to header strings."""

    def test_render_orders_read_id_then_passthrough_then_tags(self) -> None:
        """Render emits read id, then passthrough, then tags in insertion order."""
        annotation = ReadAnnotation.parse("read1 1:N:0:ACGT BC1=42:50 UMI=ACGTAC")
        self.assertEqual(annotation.render(), "read1 1:N:0:ACGT BC1=42:50 UMI=ACGTAC")

    def test_render_places_passthrough_before_tags_regardless_of_input_order(self) -> None:
        """Passthrough tokens always render before tag tokens."""
        annotation = ReadAnnotation.parse("read1 BC=1 1:N:0:ACGT")
        self.assertEqual(annotation.render(), "read1 1:N:0:ACGT BC=1")

    def test_render_preserves_tag_insertion_order(self) -> None:
        """Tags render in the order they were first inserted."""
        annotation = ReadAnnotation.parse("read1 UMI=ACGT BC=42")
        self.assertEqual(annotation.render(), "read1 UMI=ACGT BC=42")

    def test_render_read_id_only_returns_read_id(self) -> None:
        """A read id with no metadata renders to just the read id."""
        annotation = ReadAnnotation.parse("read1")
        self.assertEqual(annotation.render(), "read1")

    def test_render_empty_annotation_returns_empty_string(self) -> None:
        """An empty annotation renders to an empty string."""
        annotation = ReadAnnotation.parse("")
        self.assertEqual(annotation.render(), "")


class TestReadAnnotationRoundTrip(unittest.TestCase):
    """Round-trip stability of parse followed by render."""

    def test_parse_render_round_trip_for_various_headers(self) -> None:
        """Parsing a rendered annotation reproduces the original annotation."""
        headers = [
            "read1 BC=42 UMI=ACGT",
            "read1",
            "read1 1:N:0:ACGT BC1=42:50 UMI=ACGTAC",
            ILLUMINA_HEADER,
            "",
        ]
        for header in headers:
            with self.subTest(header=header):
                annotation = ReadAnnotation.parse(header)
                self.assertEqual(ReadAnnotation.parse(annotation.render()), annotation)


class TestReadAnnotationGet(unittest.TestCase):
    """Lookup of tag values."""

    def test_get_returns_value_for_present_key(self) -> None:
        """Get returns the stored value for a present key."""
        annotation = ReadAnnotation.parse("read1 BC=42")
        self.assertEqual(annotation.get("BC"), "42")

    def test_get_returns_none_for_missing_key(self) -> None:
        """Get returns None for a key that is not present."""
        annotation = ReadAnnotation.parse("read1 BC=42")
        self.assertIsNone(annotation.get("UMI"))


class TestReadAnnotationSet(unittest.TestCase):
    """In-place mutation of tag values."""

    def test_set_adds_new_key(self) -> None:
        """Set adds a new tag that is then retrievable and rendered."""
        annotation = ReadAnnotation.parse("read1")
        annotation.set("BC", "42")
        self.assertEqual(annotation.get("BC"), "42")
        self.assertEqual(annotation.tags, {"BC": "42"})
        self.assertEqual(annotation.render(), "read1 BC=42")

    def test_set_overwrites_existing_key_in_place(self) -> None:
        """Set overwrites the value of an existing key in place."""
        annotation = ReadAnnotation.parse("read1 BC=42")
        annotation.set("BC", "99")
        self.assertEqual(annotation.get("BC"), "99")
        self.assertEqual(annotation.tags, {"BC": "99"})


class TestReadAnnotationEquality(unittest.TestCase):
    """Value-object equality over read id, tags, and passthrough."""

    def test_equal_headers_parse_to_equal_annotations(self) -> None:
        """Two independently parsed equal headers compare equal."""
        left = ReadAnnotation.parse("read1 1:N:0:ACGT BC=42 UMI=ACGT")
        right = ReadAnnotation.parse("read1 1:N:0:ACGT BC=42 UMI=ACGT")
        self.assertEqual(left, right)

    def test_annotations_differing_in_read_id_are_not_equal(self) -> None:
        """Annotations with different read ids compare unequal."""
        left = ReadAnnotation.parse("read1 BC=42")
        right = ReadAnnotation.parse("read2 BC=42")
        self.assertNotEqual(left, right)

    def test_annotations_differing_in_tag_value_are_not_equal(self) -> None:
        """Annotations with different tag values compare unequal."""
        left = ReadAnnotation.parse("read1 BC=42")
        right = ReadAnnotation.parse("read1 BC=99")
        self.assertNotEqual(left, right)

    def test_annotations_differing_in_passthrough_are_not_equal(self) -> None:
        """Annotations with different passthrough tokens compare unequal."""
        left = ReadAnnotation.parse("read1 1:N:0:ACGT")
        right = ReadAnnotation.parse("read1 2:N:0:ACGT")
        self.assertNotEqual(left, right)


class TestReadAnnotationModulePurity(unittest.TestCase):
    """Import-layer purity of the read-annotation module (AC3)."""

    def test_module_does_not_import_barcode_or_chemistry(self) -> None:
        """The io module must not depend on barcode or chemistry packages."""
        try:
            import carmack.io.read_annotation as read_annotation_module
        except ImportError as exc:
            self.fail(f"carmack.io.read_annotation is not importable: {exc}")

        source = Path(read_annotation_module.__file__).read_text(encoding="utf-8")
        imported = collect_imported_module_names(source)
        for name in imported:
            with self.subTest(imported=name):
                self.assertFalse(
                    name.startswith("carmack.barcode"),
                    f"io module must not import from carmack.barcode: {name}",
                )
                self.assertFalse(
                    name.startswith("carmack.chemistry"),
                    f"io module must not import from carmack.chemistry: {name}",
                )


if __name__ == "__main__":
    unittest.main()

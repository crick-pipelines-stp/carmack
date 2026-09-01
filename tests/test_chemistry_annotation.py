"""Unit tests for the chemistry annotation helpers (ST2 of issue #14)."""

import ast
import unittest
from pathlib import Path

from carmack.chemistry.annotation import POS_SUFFIX, format_span, parse_span, position_key
from carmack.chemistry.read_component import ReadComponent


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


class TestPositionKey(unittest.TestCase):
    """Position-key derivation from component names."""

    def test_pos_suffix_is_expected_constant(self) -> None:
        """The module exposes the canonical position suffix constant."""
        self.assertEqual(POS_SUFFIX, "_POS")

    def test_position_key_appends_suffix_for_various_names(self) -> None:
        """position_key appends the suffix to the component name."""
        cases = [("BC1", "BC1_POS"), ("UMI", "UMI_POS")]
        for name, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(position_key(name), expected)


class TestFormatSpan(unittest.TestCase):
    """Rendering of 0-based half-open spans to text."""

    def test_format_span_renders_start_colon_end(self) -> None:
        """A span renders as ``start:end`` for the 8bp-BC1-at-42 example."""
        self.assertEqual(format_span(42, 50), "42:50")

    def test_format_span_zero_length_span_is_valid(self) -> None:
        """A zero-length span where start equals end renders successfully."""
        self.assertEqual(format_span(42, 42), "42:42")

    def test_format_span_negative_start_raises_value_error(self) -> None:
        """A negative start position is rejected."""
        with self.assertRaises(ValueError):
            format_span(-1, 5)

    def test_format_span_end_before_start_raises_value_error(self) -> None:
        """An end position before the start is rejected."""
        with self.assertRaises(ValueError):
            format_span(5, 3)


class TestParseSpan(unittest.TestCase):
    """Parsing of 0-based half-open span text back to bounds."""

    def test_parse_span_returns_tuple_of_two_ints(self) -> None:
        """A well-formed span parses to a two-int tuple."""
        result = parse_span("42:50")
        self.assertEqual(result, (42, 50))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)

    def test_parse_span_zero_length_span_is_valid(self) -> None:
        """A zero-length span parses successfully."""
        self.assertEqual(parse_span("42:42"), (42, 42))

    def test_parse_span_is_inverse_of_format_span(self) -> None:
        """parse_span and format_span round-trip in both directions."""
        pairs = [(0, 0), (0, 8), (42, 50)]
        for start, end in pairs:
            with self.subTest(start=start, end=end):
                text = format_span(start, end)
                self.assertEqual(parse_span(text), (start, end))
                self.assertEqual(format_span(*parse_span(text)), text)

    def test_parse_span_malformed_input_raises_value_error(self) -> None:
        """Malformed or out-of-bounds span text is rejected."""
        malformed = ["", "abc", "1", "1:2:3", "1:", ":2", "-1:2", "1:-2", "5:3", "4.2:5"]
        for text in malformed:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_span(text)


class TestReadComponentPositionKey(unittest.TestCase):
    """Position-key convenience property on ReadComponent."""

    def test_read_component_position_key_matches_free_function(self) -> None:
        """The property mirrors position_key applied to the component name."""
        cases = ["BC1", "UMI"]
        for name in cases:
            with self.subTest(name=name):
                component = ReadComponent(name=name, length=8)
                self.assertEqual(component.position_key, f"{name}_POS")
                self.assertEqual(component.position_key, position_key(name))


class TestAnnotationModulePurity(unittest.TestCase):
    """Import-layer purity of the chemistry annotation module."""

    def test_module_does_not_import_from_io(self) -> None:
        """The chemistry module must not depend on the io package."""
        try:
            import carmack.chemistry.annotation as annotation_module
        except ImportError as exc:
            self.fail(f"carmack.chemistry.annotation is not importable: {exc}")

        module_file = getattr(annotation_module, "__file__", None)
        if module_file is None:
            self.fail("carmack.chemistry.annotation has no __file__ to inspect")

        source = Path(module_file).read_text(encoding="utf-8")
        imported = collect_imported_module_names(source)
        for name in imported:
            with self.subTest(imported=name):
                self.assertFalse(
                    name.startswith("carmack.io"),
                    f"chemistry module must not import from carmack.io: {name}",
                )


if __name__ == "__main__":
    unittest.main()

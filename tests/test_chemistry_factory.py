# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest

from carmack.chemistry.chemistry_factory import ChemistryFactory


class TestChemistryFactory(unittest.TestCase):
    def test_chem_factory_class_noinit(self):
        """Test the class cant be instantiated."""

        with self.assertRaises(TypeError):
            ChemistryFactory()

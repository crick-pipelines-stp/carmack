from carmack.chemistry.chemistry_factory import ChemistryFactory

def test_class_noinit(self):
    """Test the class cant be instantiated."""

    with self.assertRaises(TypeError):
        ChemistryFactory()
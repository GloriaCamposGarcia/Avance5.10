import unittest
from src.normalization import strip_accents, normalize_whitespace, normalize_entity_name

class TestNormalization(unittest.TestCase):
    def test_strip_accents(self):
        self.assertEqual(strip_accents("México"), "Mexico")
        self.assertEqual(strip_accents("Logística"), "Logistica")
        self.assertEqual(strip_accents("áéíóúÁÉÍÓÚñÑ"), "aeiouAEIOUnN")

    def test_normalize_whitespace(self):
        self.assertEqual(normalize_whitespace("   hola    mundo   "), "hola mundo")
        self.assertEqual(normalize_whitespace("\n\t  hola  \t  \n"), "hola")

    def test_normalize_entity_name_moral(self):
        res = normalize_entity_name("Logistica Retail S.A.", "MORAL")
        self.assertEqual(res['name_raw'], "Logistica Retail S.A.")
        self.assertEqual(res['name_norm'], "logistica retail s.a")
        self.assertEqual(res['name_base'], "logistica retail")
        self.assertEqual(res['name_initials'], "lr")
        self.assertEqual(res['name_token_count'], 2)
        self.assertEqual(res['name_char_count'], 15)
        self.assertEqual(res['removed_legal_suffix_pattern_count'], 1)

    def test_normalize_entity_name_physical(self):
        res = normalize_entity_name("David Smith", "FISICA")
        self.assertEqual(res['name_raw'], "David Smith")
        self.assertEqual(res['name_norm'], "david smith")
        self.assertEqual(res['name_base'], "david smith")
        self.assertEqual(res['name_initials'], "ds")
        self.assertEqual(res['name_token_count'], 2)
        self.assertEqual(res['name_char_count'], 10)
        self.assertEqual(res['removed_legal_suffix_pattern_count'], 0)

if __name__ == '__main__':
    unittest.main()

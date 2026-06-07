import unittest
from src.variants import generate_name_variants

class TestVariants(unittest.TestCase):
    def test_empty_name(self):
        self.assertEqual(generate_name_variants("", "FISICA"), [])
        self.assertEqual(generate_name_variants("   ", "MORAL"), [])

    def test_single_token(self):
        res = generate_name_variants("logistica", "MORAL")
        variants = {r['variant'] for r in res}
        self.assertIn("logistica", variants)

    def test_multiple_tokens(self):
        res = generate_name_variants("logistica retail", "MORAL")
        variants = {r['variant']: r['rule'] for r in res}
        
        self.assertIn("logistica retail", variants)
        self.assertEqual(variants["logistica retail"], "base_canonical")
        
        self.assertIn("retail logistica", variants)
        self.assertEqual(variants["retail logistica"], "reverse_order")
        
        self.assertIn("l r", variants)
        self.assertEqual(variants["l r"], "initials_full")
        
        self.assertIn("l retail", variants)
        self.assertEqual(variants["l retail"], "first_initial_last")

    def test_moral_noise_removal(self):
        res = generate_name_variants("apex trading", "MORAL")
        variants = {r['variant']: r['rule'] for r in res}
        
        self.assertIn("apex", variants)
        self.assertEqual(variants["apex"], "moral_remove_corp_noise")

if __name__ == '__main__':
    unittest.main()

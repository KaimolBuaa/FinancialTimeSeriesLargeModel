import unittest


class ProxyFactorRegistryTests(unittest.TestCase):
    def test_registry_has_exact_stable_contract(self):
        from factorpanel_data.proxy_registry import (
            FACTOR_WINDOWS,
            build_proxy_factor_registry,
        )

        factors = build_proxy_factor_registry()

        self.assertEqual(FACTOR_WINDOWS, (2, 3, 5, 10, 20, 30, 60, 120))
        self.assertEqual(len(factors), 128)
        self.assertEqual(len({item.name for item in factors}), 128)
        self.assertEqual(
            [item.name for item in factors[:3]],
            ["pf_kmid", "pf_klen", "pf_kmid2"],
        )
        self.assertEqual(factors[-1].name, "pf_vstd_120")
        self.assertTrue(
            all(",-" not in item.expression.replace(" ", "") for item in factors)
        )

    def test_labels_are_separate_and_forward_looking(self):
        from factorpanel_data.proxy_registry import build_label_registry

        labels = build_label_registry()

        self.assertEqual(
            [item.name for item in labels],
            ["ret_1d", "ret_5d", "ret_20d"],
        )
        self.assertEqual([item.horizon for item in labels], [1, 5, 20])
        self.assertTrue(
            all(
                "Ref($close,-" in item.expression.replace(" ", "")
                for item in labels
            )
        )

    def test_definitions_are_immutable(self):
        from dataclasses import FrozenInstanceError

        from factorpanel_data.proxy_registry import build_proxy_factor_registry

        factor = build_proxy_factor_registry()[0]
        with self.assertRaises(FrozenInstanceError):
            factor.name = "changed"

    def test_two_day_rsquared_uses_three_observation_minimum(self):
        from factorpanel_data.proxy_registry import build_proxy_factor_registry

        factors = {item.name: item for item in build_proxy_factor_registry()}

        self.assertEqual(factors["pf_rsqr_2"].expression, "Rsquare($close,3)")


if __name__ == "__main__":
    unittest.main()

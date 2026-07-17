from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorpanel_fm.losses import (
    masked_huber_loss,
    negative_cross_sectional_ic_loss,
    pairwise_logistic_loss,
    quantile_pinball_loss,
)


class MaskedLossTests(unittest.TestCase):
    def test_masked_huber_uses_only_masked_finite_values(self) -> None:
        prediction = torch.tensor([0.0, 2.0, 10.0, float("nan")], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 0.0, 1.0])

        loss = masked_huber_loss(prediction, target, torch.ones(4, dtype=torch.bool))

        torch.testing.assert_close(loss, torch.tensor(11.5 / 3.0))
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_masked_huber_empty_selection_is_graph_connected_zero(self) -> None:
        prediction = torch.randn(3, requires_grad=True)

        loss = masked_huber_loss(
            prediction,
            torch.full((3,), float("nan")),
            torch.zeros(3, dtype=torch.bool),
        )

        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_quantile_pinball_matches_definition_and_backpropagates(self) -> None:
        prediction = torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True)
        target = torch.tensor([[1.0]])

        loss = quantile_pinball_loss(
            prediction,
            target,
            torch.tensor([[True]]),
        )

        torch.testing.assert_close(loss, torch.tensor(1.0 / 15.0))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_quantile_pinball_validates_shapes_and_handles_empty_mask(self) -> None:
        prediction = torch.randn(2, 3, 3, requires_grad=True)
        target = torch.randn(2, 3)
        mask = torch.zeros(2, 3, dtype=torch.bool)

        loss = quantile_pinball_loss(prediction, target, mask)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
        with self.assertRaises(ValueError):
            quantile_pinball_loss(prediction[..., :2], target, mask)
        with self.assertRaises(ValueError):
            quantile_pinball_loss(prediction, target[:, :2], mask)


class CrossSectionalLossTests(unittest.TestCase):
    def test_ic_averages_only_valid_nonconstant_cross_sections(self) -> None:
        scores = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]],
            requires_grad=True,
        )
        targets = torch.tensor([[[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]])

        loss = negative_cross_sectional_ic_loss(
            scores,
            targets,
            torch.ones_like(scores, dtype=torch.bool),
        )

        torch.testing.assert_close(loss, torch.tensor(1.0), atol=1e-6, rtol=1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_ic_skips_short_and_constant_sections_and_returns_safe_zero(self) -> None:
        scores = torch.ones(2, 3, 2, requires_grad=True)
        targets = torch.randn_like(scores)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[:, 0] = True

        loss = negative_cross_sectional_ic_loss(scores, targets, mask)

        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_pairwise_logistic_matches_all_unordered_pairs(self) -> None:
        scores = torch.tensor([[[0.0], [1.0], [2.0]]], requires_grad=True)
        targets = torch.tensor([[[0.0], [1.0], [2.0]]])

        loss = pairwise_logistic_loss(
            scores,
            targets,
            torch.ones_like(scores, dtype=torch.bool),
            max_pairs=10,
        )

        expected = (2.0 * F.softplus(torch.tensor(-1.0)) + F.softplus(torch.tensor(-2.0))) / 3.0
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_pairwise_sampling_is_reproducible_and_empty_is_safe(self) -> None:
        scores = torch.linspace(-2.0, 2.0, 30).reshape(1, 30, 1).requires_grad_()
        targets = torch.rand(1, 30, 1)
        mask = torch.ones_like(scores, dtype=torch.bool)

        first = pairwise_logistic_loss(
            scores,
            targets,
            mask,
            max_pairs=7,
            generator=torch.Generator().manual_seed(91),
        )
        second = pairwise_logistic_loss(
            scores,
            targets,
            mask,
            max_pairs=7,
            generator=torch.Generator().manual_seed(91),
        )
        torch.testing.assert_close(first, second)

        empty = pairwise_logistic_loss(
            scores,
            torch.ones_like(targets),
            mask,
            generator=torch.Generator().manual_seed(4),
        )
        self.assertEqual(empty.item(), 0.0)
        empty.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_pair_limit_applies_independently_to_each_cross_section(self) -> None:
        scores = torch.tensor(
            [[[0.0], [1.0], [4.0]], [[0.0], [2.0], [3.0]]],
        )
        targets = torch.tensor(
            [[[0.0], [1.0], [2.0]], [[2.0], [1.0], [0.0]]],
        )
        mask = torch.ones_like(scores, dtype=torch.bool)

        limited = pairwise_logistic_loss(
            scores,
            targets,
            mask,
            max_pairs=3,
            generator=torch.Generator().manual_seed(9),
        )
        all_pairs = pairwise_logistic_loss(scores, targets, mask, max_pairs=6)

        torch.testing.assert_close(limited, all_pairs)

    def test_cross_sectional_losses_validate_shapes(self) -> None:
        scores = torch.randn(2, 3, 4)
        target = torch.randn_like(scores)
        mask = torch.ones_like(scores, dtype=torch.bool)

        with self.assertRaises(ValueError):
            negative_cross_sectional_ic_loss(scores[:, :, 0], target[:, :, 0], mask[:, :, 0])
        with self.assertRaises(ValueError):
            pairwise_logistic_loss(scores, target[:, :2], mask)
        with self.assertRaises(ValueError):
            pairwise_logistic_loss(scores, target, mask, max_pairs=0)


if __name__ == "__main__":
    unittest.main()

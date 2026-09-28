#!/usr/bin/env python3
"""
Tests for util/training.py: the opponent split used by sb-train.py's selfplay-edax mode.
"""

import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.training import parse_edax_mix, mixed_block_plan


class TestTrainingMix(unittest.TestCase):

    def test_parse_edax_mix(self):
        self.assertEqual(parse_edax_mix("1, 2,3", "0.1,0.1, 0.2"), ([1, 2, 3], [0.1, 0.1, 0.2]))
        with self.assertRaises(ValueError):
            parse_edax_mix("1,2", "0.5")
        with self.assertRaises(ValueError):
            parse_edax_mix("1,x", "0.5,0.5")

    def test_plan_order_and_total(self):
        plan = mixed_block_plan(100_000, 0.6, [1, 2, 3], [0.1, 0.1, 0.1], 0.1)
        self.assertEqual([(o, d) for o, d, _ in plan],
                         [("Self", None), ("Edax", 1), ("Edax", 2), ("Edax", 3), ("Random", None)])
        self.assertEqual(sum(t for _, _, t in plan), 100_000)
        self.assertEqual(plan[0][2], 60_000)
        self.assertEqual(plan[1][2], 10_000)

    def test_ratios_are_normalized(self):
        a = mixed_block_plan(10_000, 6, [2], [3], 1)
        b = mixed_block_plan(10_000, 0.6, [2], [0.3], 0.1)
        self.assertEqual(a, b)

    def test_rounding_leftover_keeps_total_exact(self):
        plan = mixed_block_plan(99_999, 1, [1, 2], [1, 1], 1)
        self.assertEqual(sum(t for _, _, t in plan), 99_999)

    def test_zero_ratios_dropped(self):
        plan = mixed_block_plan(1_000, 0.7, [1, 2], [0.3, 0.0], 0.0)
        self.assertEqual([(o, d) for o, d, _ in plan], [("Self", None), ("Edax", 1)])
        self.assertEqual(sum(t for _, _, t in plan), 1_000)

    def test_all_zero_raises(self):
        with self.assertRaises(ValueError):
            mixed_block_plan(1_000, 0, [1], [0], 0)


if __name__ == '__main__':
    unittest.main()

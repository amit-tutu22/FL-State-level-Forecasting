"""Regression tests for audited PDF extraction."""

# Test names describe assertions; repetitive test docstrings add no context.
# pylint: disable=missing-function-docstring,missing-class-docstring,duplicate-code
import unittest

from data_pipeline.pdf import number, parse_table


class ExtractionTests(unittest.TestCase):
    def table(self):
        return [
            ["State/UT", "Sep-25", "Sep-26", "Growth (%)"],
            ["Bihar", "100", "110", "10%"],
            ["Other Territory", "20", "22", "10%"],
            ["Grand Total", "120", "132", "10%"],
        ]

    def test_split_cells_and_non_clients(self):
        table = [[None, *r, "", None] for r in self.table()]
        rows, _ = parse_table(table, "2026-09")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["reference_month"], "2026-09")
        self.assertEqual(rows[2]["row_role"], "non_client")

    def test_cumulative_header_rejected(self):
        table = self.table()
        table[0][2] = "2026-27"
        with self.assertRaises(ValueError):
            parse_table(table, "2026-09")

    def test_wrong_reference_rejected(self):
        with self.assertRaises(ValueError):
            parse_table(self.table(), "2026-08")

    def test_duplicate_rejected(self):
        table = self.table()
        table.insert(2, table[1])
        with self.assertRaises(ValueError):
            parse_table(table, "2026-09")

    def test_wrong_total_rejected(self):
        table = self.table()
        table[-1][2] = "999"
        with self.assertRaises(ValueError):
            parse_table(table, "2026-09")

    def test_missing_or_invalid_amount_rejected(self):
        for value in ["", "-", "NaN", "1,2,3", "-20"]:
            with self.assertRaises(ValueError):
                number(value)
        self.assertEqual(str(number("1 ,37,996")), "137996")


if __name__ == "__main__":
    unittest.main()

"""Conservative adapters for GST single-page and multi-page state-growth tables.

Requires pdfplumber==0.11.9. Unknown layouts fail with a raw-table audit.
This is not a universal historical-PDF parser or a complete dataset downloader.
"""

import argparse
import calendar
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pdfplumber

from .geography import CANONICAL

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
NON_CLIENTS = {"Other Territory", "Center Jurisdiction", "Centre Jurisdiction"}
STATES = set(CANONICAL) - {
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Dadra and Nagar Haveli and Daman and Diu",
}
STATES.add("Dadra and Nagar Haveli and Daman & Diu")


def clean(value):
    """Normalize whitespace in an extracted PDF cell."""
    return " ".join((value or "").split())


def compact(row):
    """Remove blank fragmentation cells without shifting meaningful values."""
    return [clean(c) for c in row if clean(c)]


def number(value):
    """Parse validated Indian or western numeric groupings as decimals."""
    raw = re.sub(r"\s", "", value)
    # Validate grouping before removing separators (Indian and western forms).
    if not re.fullmatch(
        r"(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?", raw
    ):
        raise ValueError(f"Invalid nonnegative monetary value: {value!r}")
    return Decimal(raw.replace(",", ""))


def period(value):
    """Parse a three-letter month and a two- or four-digit year."""
    match = re.fullmatch(r"([A-Za-z]{3})-(\d{2}|\d{4})", value)
    if not match or match[1].lower() not in MONTHS:
        raise ValueError(f"Not a monthly header: {value!r}")
    year = int(match[2])
    if year < 100:
        year += 2000
    return f"{year:04d}-{MONTHS[match[1].lower()]:02d}"


# Keep source-schema validation/provenance together for auditability.
# pylint: disable-next=too-many-locals,too-many-branches
def parse_table(table, expected_period):
    """Validate monthly columns, geography rows and rounded printed totals."""
    rows = [compact(r) for r in table]
    headers = [i for i, r in enumerate(rows) if r and r[0] == "State/UT"]
    if len(headers) != 1:
        raise ValueError("Expected exactly one State/UT header")
    start = headers[0]
    header = rows[start]
    if len(header) != 4 or header[3] != "Growth (%)":
        raise ValueError(f"Unsupported column layout: {header}")
    previous, current = period(header[1]), period(header[2])
    if (
        current != expected_period
        or previous != f"{int(current[:4])-1:04d}{current[4:]}"
    ):
        raise ValueError("Header periods do not match requested month and prior year")
    observations, totals, seen = [], {}, set()
    for row in rows[start + 1 :]:
        if not row:
            continue
        if len(row) != 4:
            raise ValueError(f"Unexpected row width: {row}")
        state, prior, latest, growth = row
        if state in seen:
            raise ValueError(f"Duplicate geography: {state}")
        seen.add(state)
        if not re.fullmatch(r"-?\d+(?:\.\d+)?%", growth):
            raise ValueError(f"Unexpected growth entry: {growth}")
        values = [number(prior), number(latest)]
        if state == "Grand Total":
            totals = dict(zip([previous, current], values))
            continue
        if state not in STATES | NON_CLIENTS:
            raise ValueError(f"Unrecognized geography; review mapping: {state}")
        for month, value in zip([previous, current], values):
            observations.append(
                {
                    "state_source": state,
                    "reference_month": month,
                    "amount_inr_crore": str(value),
                    "row_role": "non_client" if state in NON_CLIENTS else "state_or_ut",
                    "period_role": (
                        "current" if month == current else "prior_year_comparison"
                    ),
                }
            )
    if not totals or not observations:
        raise ValueError("Missing rows or grand total")
    checks = []
    for month, published in totals.items():
        selected = [r for r in observations if r["reference_month"] == month]
        summed = sum(Decimal(r["amount_inr_crore"]) for r in selected)
        # Individual printed values and total are rounded to whole crore.
        tolerance = Decimal(len(selected) + 1) / 2
        if abs(summed - published) > tolerance:
            raise ValueError(
                f"Total reconciliation failed for {month}: {summed} vs {published}"
            )
        checks.append(
            {
                "month": month,
                "sum_printed_rows": str(summed),
                "published_total": str(published),
                "rounding_tolerance_crore": str(tolerance),
            }
        )
    return observations, checks


# Keep source-schema validation/provenance together for auditability.
# pylint: disable-next=too-many-branches
def collect_monthly_table(pages, expected_period):
    """Assemble the single-page 2026 and continued PIB 2023 grid layouts.

    Selection requires the exact month pair and 4-column growth header, so the
    neighboring 7-column settlement tables cannot enter the extraction.
    """
    text = "\n".join(page["text"] for page in pages)
    if not re.search(r"State-wise growth of GST Revenues during", text, re.I):
        raise ValueError("Missing supported monthly state-growth heading")
    assembled, row_pages = [], {}
    started = complete = False
    for page in pages:
        for table in page["tables"]:
            rows = [compact(row) for row in table]
            if not rows:
                continue
            if not started:
                first = rows[0]
                if len(first) != 4 or first[0] != "State/UT":
                    continue
                if period(first[2]) != expected_period:
                    continue
                assembled.append(first)
                rows = rows[1:]
                started = True
            if complete:
                continue
            for row in rows:
                if not row:
                    continue
                # Verified 2023 PIB cell-border error: the final 'u' of Diu
                # falls into the amount cell. Restore only this exact pattern.
                if (
                    len(row) == 4
                    and row[0] == "Dadra and Nagar Haveli and Daman & Di"
                    and re.fullmatch(r"u\d[\d,]*", row[1])
                ):
                    row = [row[0] + "u", row[1][1:], *row[2:]]
                if (
                    len(row) == 4
                    and row[0] == "Dadra and Nagar Haveli and Daman"
                    and re.fullmatch(r"& Diu \d[\d,]*", row[1])
                ):
                    row = [row[0] + " & Diu", row[1].removeprefix("& Diu "), *row[2:]]
                if len(row) != 4:
                    raise ValueError(
                        "Unexpected continuation layout before grand total"
                    )
                assembled.append(row)
                row_pages[row[0]] = page["page"]
                if row[0] == "Grand Total":
                    complete = True
                    break
    if not complete:
        raise ValueError("Missing complete monthly state table")
    return assembled, row_pages


# Keep source-schema validation/provenance together for auditability.
# pylint: disable-next=too-many-locals
def extract(pdf_path, reference_month, source_url, output_dir):
    """Extract supported monthly tables and save CSVs with a raw source audit."""
    datetime.strptime(reference_month, "%Y-%m")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prevent a failed retry from leaving apparently valid output from an earlier run.
    if any(output_dir.iterdir()):
        raise ValueError("Use a new, empty output directory for each extraction")
    raw = pdf_path.read_bytes()
    audit = {
        "source_url": source_url,
        "source_filename": pdf_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdfplumber_version": pdfplumber.__version__,
        "pages": [],
        "status": "needs_review",
    }
    try:
        with pdfplumber.open(pdf_path) as doc:
            for index, page in enumerate(doc.pages, 1):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                audit["pages"].append({"page": index, "text": text, "tables": tables})
            table, row_pages = collect_monthly_table(audit["pages"], reference_month)
            rows, checks = parse_table(table, reference_month)
            import_exclusion_verified = (
                "Does not include GST on import of goods"
                in "\n".join(p["text"] for p in audit["pages"])
            )
            for row in rows:
                row.update(
                    {
                        "measure": (
                            "state_gst_collections_excluding_goods_imports"
                            if import_exclusion_verified
                            else "state_gst_collections_import_scope_unverified"
                        ),
                        "source_page": row_pages[row["state_source"]],
                        "source_url": source_url,
                        "source_sha256": audit["sha256"],
                        "release_reference_month": reference_month,
                        "availability_date": "",
                        "cess_treatment": "requires_historical_definition_review",
                    }
                )
            audit.update(
                {
                    "status": "extracted_definition_review_required",
                    "total_checks": checks,
                    "limitations": [
                        (
                            "Verified single-page and continued PIB monthly "
                            "state-growth grids supported."
                        ),
                        (
                            "Prior-year comparison values are later-release "
                            "vintages, not original releases."
                        ),
                        (
                            "Publication date, cess comparability and "
                            "historical coverage require review."
                        ),
                    ],
                }
            )
            fields = list(rows[0])
            for filename, selected in [
                ("observations_all.csv", rows),
                (
                    "state_current_month.csv",
                    [
                        r
                        for r in rows
                        if r["row_role"] == "state_or_ut"
                        and r["period_role"] == "current"
                    ],
                ),
            ]:
                with (output_dir / filename).open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=fields, lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(selected)
    except Exception as exc:
        audit["error"] = str(exc)
        raise
    finally:
        (output_dir / "extraction_audit.json").write_text(json.dumps(audit, indent=2))
    return rows


def main():
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--reference-month", required=True, help="YYYY-MM")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.pdf, args.reference_month, args.source_url, args.output_dir)
    print(
        f"Extracted {len(result)} observations; inspect audit and definitions before training."
    )


if __name__ == "__main__":
    main()

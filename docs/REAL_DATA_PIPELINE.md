# Real state-level GST data pipeline

## Scope and sources

The research pipeline uses observed public data. The old `data-generator/` and `fl_experiment/data/state_budget_gst.csv` are legacy synthetic prototypes for software smoke tests, not thesis evidence. The new pipeline does not import either. The Flower experiments in `fl_experiment/` read its panel output through `data_pipeline.clients.load_clients`.

Sources, accessed 29 October 2026 (India time):

- [Official GST statistics exports](https://www.gst.gov.in/download/gststatistics): 10 annual gross-collection workbooks (FY2017–18 to FY2026–27), and 9 E-Way Bill workbooks (FY2018–19 to FY2026–27). URLs are discovered from the actual index, not fabricated from filenames.
- [MoSPI CPI API](https://api.mospi.gov.in/): Combined-sector General index, base 2012 Current, base 2024 Current and base 2024 Back. Metadata is fetched first; every response page and its hash are preserved. [Official API reference implementation/schema](https://github.com/nso-india/esankhyiki-mcp/tree/main/swagger) identifies the endpoints and codes. No account is required for the requests used here.
- [GST Council archive](https://gstcouncil.gov.in/gst-revenue): PDF validation samples for September/December 2023, plus the official September 2026 GST Portal PDF.

Dataful exports require sign-in. The official workbooks provide the acquisition route used here, so Dataful credentials are unnecessary. Raw government downloads are retained locally with attribution and original notices. Bulk raw/processed exports are ignored by Git; reproduce them using the commands below. The repository includes audit metadata and limited official regression fixtures with source URLs and hashes.

## Reproduce

Use Python 3.11+ and a system `curl` executable with HTTPS certificate verification enabled. `scripts/setup_venv.sh` creates `.venv` with [uv](https://docs.astral.sh/uv/) when available, otherwise with a working local `python3.11`+.

```sh
scripts/setup_venv.sh
. .venv/bin/activate
python -m data_pipeline.acquire --output data/raw/my-vintage
python -m data_pipeline.normalize --raw data/raw/my-vintage --output data/processed/my-vintage
python -m data_pipeline.panel --processed data/processed/my-vintage --output data/reports/strict --cpi-base 2012
python -m unittest discover -s tests -v
```

Acquisition/normalization require new empty output directories. A failed download can remain as evidence; use a new directory to retry. Normalization rejects incomplete CPI pagination, missing linked workbooks, changed hashes, duplicate keys, unrecognized schema/units, code-name mismatches and failed GST reconciliation. A failed output directory is not a complete dataset.

The processed directory contains `gst.csv`, `ewb.csv`, `cpi.csv`, `state_mapping_audit.csv` and `reconciliation.json`. Missing observations stay missing. No target interpolation, synthetic replacement, train/test scaling or fitting occurs here.

CI runs these offline tests in `.github/workflows/data-pipeline.yml`, and `.github/workflows/pylint.yml` lints `data_pipeline/` and `tests/` on Python 3.11. `fl_experiment/` is linted and tested separately in `.github/workflows/fl-smoke.yml`; the legacy `data-generator/` is not linted.

## Definitions that affect the thesis

### Gross GST

The source worksheet is **State Wise Domestic GST Collection**. It excludes the separate All-India imports section and is not state receipts after IGST settlement. Preserve CGST, SGST, IGST, cess (when supplied) and the published total. Ignore the fiscal-year cumulative block and refund worksheets.

The workbook total includes cess through FY2024–25; the FY2025–26 and FY2026–27 layouts contain CGST, SGST and IGST without a cess column. Consequently, `total_crore` is the unaltered published outcome with `cess_in_published_total` metadata. It must not be treated as an automatically consistent historical target.

For a comparable outcome, the panel explicitly uses `target_gst_excluding_cess_crore = CGST + SGST + IGST`, always domestic and gross of refunds. This is separately named, not a silent redefinition of published totals. A thesis using cess-inclusive collections must restrict or reconstruct a consistent period before modelling. Do not infer zero cess from an absent column.

### E-Way Bills

Keep intra-state, inter-state outward and inter-state inward flows separate. Each has supplier count, bill count and assessable value in INR crore. The main panel includes the six bill-count/value features; supplier counts are also preserved in the export. Never add inward/outward flows into a purported unique national transaction count.

The earliest linked workbook actually starts July 2018, even though E-Way Bills existed earlier. This pipeline does not invent April–June observations. The FY2026–27 filename does not imply future months: downloaded records stop August 2026.

### CPI

Use General, Combined only. Preserve base, series, status, index and published inflation. CPI 2012 and 2024 remain separate series and separate panel runs. The fetched 2024 back series contains All-India observations only; it cannot be assigned to states. The 2012 state series has missing April–May 2020 observations. Do not fill these with future observations or imply a continuous state 2024-base backcast.

## Geography

GST codes and CPI codes use different namespaces; CPI codes also change between bases. Source metadata/code-name checks and explicit aliases map names to internal IDs. Unrecognized names are not fuzzy-matched.

- J&K before October 2019 uses an undivided ID. October 2019 is quarantined. Separate post-split IDs are used where supported. The old-base CPI J&K series after the split is quarantined because comparable boundary coverage has not been established. Pre-formation Ladakh cells are not observations for a new client.
- Legacy Daman & Diu / Dadra & Nagar Haveli labels after January 2020 are quarantined. The workbook labels and codes do not justify automatically mapping or summing them into the merged territory. CPI is an index and cannot be summed even if monetary series can sometimes be aggregated.
- All India, Other Territory, CBIC/OIDAR and Center Jurisdiction are retained for audit/reconciliation and excluded as state clients.

Legal boundary references: [MHA J&K division](https://www.mha.gov.in/jammu-kashmir-ladakh-affairs), [Dadra/Nagar Haveli/Daman/Diu administration](https://ddd.gov.in/introduction/). These establish the transition dates, not the statistical coverage of an individual source. The exported mapping audit makes unresolved source definitions explicit.

## Availability and revisions

A reference month is not a publication date. A nominal release calendar is also not proof that a **revised value downloaded today** was known years ago.

**Strict default:** `available_at` is the retrieval date for the exact downloaded snapshot. Thus the current bulk archive produces zero complete historical forecasting rows under strict backtesting. This is intentional: it prevents revision leakage rather than manufacturing release dates. Source files do not supply a complete historical vintage archive.

To supply genuinely archived release vintages, pass `--release-registry registry.csv` with columns:

```text
source_sha256,reference_month,available_at,exact_vintage_verified,evidence_url
```

Dates use YYYY-MM-DD, and `exact_vintage_verified` must be `true`. A reviewer must verify that the cited archive supports the exact file hash/value vintage; the software validates keys, dates and the explicit assertion, not the truth of an external document. A release date taken from an unrelated report does not meet this requirement.

**Retrospective sensitivity mode:** explicitly opt in to use latest-vintage values with an assumed 45-day lag after month end:

```sh
python -m data_pipeline.panel --processed data/processed/my-vintage --output data/reports/lagged-2012 --cpi-base 2012 --mode assumed_lag_latest_vintage --lag-days 45
```

This is not a real-time backtest. Revision bias remains, and 45 days is an assumption, not a verified release schedule. Repeat at plausible lags and state the limitation in the thesis. `load_clients()` refuses this mode unless `allow_assumed_vintages=True` is explicitly supplied.

For each target month, forecast issuance is the previous calendar month's final day. Features must refer to earlier months, be available by issuance and be no more than three months old. The latest eligible reference month and then its latest eligible vintage are selected. Feature reference/availability dates and source hashes remain in the panel. Targets are evaluation labels and are never used as same-month inputs. Model training still needs chronological train/validation/test splits and fold-local preprocessing.

## PDF adapters

`python -m data_pipeline.pdf report.pdf --reference-month YYYY-MM --source-url URL --output-dir new-folder`

Validated layouts:

1. Single-page September 2026 grid with fragmented empty cells.
2. December 2023 PIB table continuing across two pages.
3. September 2023 PIB browser-print table continuing across three pages, including a verified long state name crossing a cell boundary.

The parser uses explicit monthly headers and stops at the table's grand total before settlement tables. It checks duplicates, numerical cells, geography names and totals with printed-value rounding tolerances. Only two exact, observed Diu cell-boundary artifacts are repaired. Unknown layouts fail for review. An absent imports footnote yields `import_scope_unverified`; no definition is invented. PDFs remain a separate validation/gap-fill stream and are not automatically spliced into the component workbook series.

The offline regression fixtures preserve extracted tables plus source URLs/hashes. Scanned PDFs and arbitrary historical layouts are not claimed to be supported.

## Current audit and remaining work

See [the dated audit](audits/2026-10-29.md). The full available official exports are acquired. Full original-release histories, unresolved territory definitions and state-level CPI rebasing require further evidence. No forecasting accuracy or privacy claims have been tested by this data work.

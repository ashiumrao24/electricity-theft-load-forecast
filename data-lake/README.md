# Energy Data Lake

Extends the forecasting project in this repo with a PySpark bronze, silver, and gold data lake over the same UCI dataset.

## Architecture

```text
raw text file -> bronze (raw Parquet, by year)
                   -> silver (typed, deduplicated, by year and month)  -> gold (Spark SQL tables)
                        |                                                  |
                    quarantine                                      quality gate
```

Silver is written to a staging path first, so a failed gate leaves the published layer untouched.

## Gold tables

| Table | Purpose |
|---|---|
| `hourly_energy` | Complete hourly spine with coverage, so missing hours stay visible |
| `daily_usage` | Daily energy split by sub-meter |
| `monthly_kpis` | Rolling averages and year-over-year change |
| `data_gaps` | Every stretch of missing readings, using a gaps-and-islands query |
| `hourly_anomalies` | Hours 3+ standard deviations from the same weekday and hour over the 8 prior weeks |

## Results on the full dataset

Measured on a MacBook in Spark local mode, not a cluster.

| Measure | Result |
|---|---:|
| Bronze rows | 2,075,259 |
| Silver rows | 2,049,280 |
| Quarantined rows | 25,979 |
| Rows unaccounted for | 0 |
| Full pipeline runtime | About 30 seconds |
| Hours with no readings | 421 |
| Gaps found | 71 (longest 7,226 minutes) |
| One-year query, Parquet vs raw text | About 3x faster (3.2x and 3.9x across two benchmark runs) |
| Tests | 15 passing |

`reports/` holds the latest run manifest, benchmark, and CSV exports.

## Design decisions

- **Hourly spine.** Hours without readings hold `NULL`, not 0, so an outage cannot look like zero usage.
- **Separate observed and estimated energy.** Partial hours report both the measured kWh and an estimate from average power.
- **Leak-free baselines.** The anomaly window ends at `1 PRECEDING`, and a test proves future values cannot change it.
- **Reconciliation.** Silver plus quarantine must equal bronze, and hourly energy must equal the minute-level total.
- **Overwrite modes.** Bronze uses dynamic overwrite, so a late file for one year is a safe backfill. Silver and gold are rebuilt with static overwrite, so stale partitions disappear.

## Run it

Requires Python 3.13 or earlier. Python 3.14 is not tested with Spark 3.5. Java is installed by `jdk4py`, or set `JAVA_HOME` to use your own.

```bash
cd data-lake
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py              # saves to ../data/

python -m energy_lake.pipeline
PYTHONPATH=. python scripts/benchmark.py
PYTHONPATH=. python scripts/export_reports.py
pytest
```

## Limitations

- Runs on one machine. Cluster, S3, and EMR or Glue execution are not tested.
- `data_gaps` uses a window with no partition. That is fine for one meter, but multiple meters would need `PARTITION BY meter_id`.
- Anomaly flags come from a statistical threshold on one household and have no ground truth. 1,678 hours are flagged (1,437 spikes, 241 drops), so treat them as candidates for review.

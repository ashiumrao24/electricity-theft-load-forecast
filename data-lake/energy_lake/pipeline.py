"""Run bronze -> silver -> gold, with a quality gate before each publish."""
import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from energy_lake import bronze as bronze_layer
from energy_lake import gold as gold_layer
from energy_lake import quality
from energy_lake import silver as silver_layer
from energy_lake.session import build_spark


def run_pipeline(spark, input_path: str, lake_path: str) -> dict:
    bronze_path = f"{lake_path}/bronze/readings"
    silver_path = f"{lake_path}/silver/readings"
    quarantine_path = f"{lake_path}/quarantine/readings"
    gold_path = f"{lake_path}/gold"
    timings = {}

    started = time.perf_counter()
    raw = bronze_layer.read_raw(spark, input_path)
    bronze_layer.write_bronze(bronze_layer.to_bronze(raw), bronze_path)
    bronze = spark.read.parquet(bronze_path)
    bronze_count = bronze.count()
    timings["bronze_seconds"] = round(time.perf_counter() - started, 1)

    started = time.perf_counter()
    silver, quarantine = silver_layer.to_silver(bronze)
    # Write to a staging path so that a failed gate leaves the published silver untouched.
    staging_silver = f"{lake_path}/_staging/silver"
    staging_quarantine = f"{lake_path}/_staging/quarantine"
    silver_layer.write_silver(silver, quarantine, staging_silver, staging_quarantine)
    # An empty quarantine writes no Parquet files, so its schema must be supplied on read.
    quarantine_schema = quarantine.schema
    silver = spark.read.parquet(staging_silver)
    quarantine = spark.read.schema(quarantine_schema).parquet(staging_quarantine)
    silver_checks = quality.check_silver(bronze_count, silver, quarantine)
    quality.enforce(silver_checks)
    silver_layer.write_silver(silver, quarantine, silver_path, quarantine_path)
    silver = spark.read.parquet(silver_path)
    quarantine = spark.read.schema(quarantine_schema).parquet(quarantine_path)
    timings["silver_seconds"] = round(time.perf_counter() - started, 1)

    started = time.perf_counter()
    tables = gold_layer.build_gold(spark, silver)
    gold_checks = quality.check_gold(silver, tables["hourly_energy"])
    quality.enforce(gold_checks)
    gold_counts = gold_layer.write_gold(tables, gold_path)
    timings["gold_seconds"] = round(time.perf_counter() - started, 1)

    reasons = {r["reject_reason"]: r["count"]
               for r in quarantine.groupBy("reject_reason").count().collect()}
    manifest = {
        "finished_at": datetime.now(UTC).isoformat(),
        "input": input_path,
        "row_counts": {
            "bronze": bronze_count,
            "silver": silver.count(),
            "quarantine": quarantine.count(),
            "quarantine_by_reason": reasons,
            "gold": gold_counts,
        },
        "checks": [asdict(c) for c in silver_checks + gold_checks],
        "timings": timings,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../data/household_power_consumption.txt")
    parser.add_argument("--lake", default="data/lake")
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    manifest = run_pipeline(spark, args.input, args.lake)
    spark.stop()

    output = Path("reports/run_manifest.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

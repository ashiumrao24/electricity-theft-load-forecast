import json

import pytest
from pyspark.sql import functions as F

from energy_lake import bronze as bronze_layer
from energy_lake import gold as gold_layer
from energy_lake import quality
from energy_lake import silver as silver_layer
from energy_lake.pipeline import run_pipeline
from tests.conftest import reading


def silver_from(spark, path):
    raw = bronze_layer.read_raw(spark, path)
    return silver_layer.to_silver(bronze_layer.to_bronze(raw))


def reasons(quarantine):
    return {r["reject_reason"]: r["count"]
            for r in quarantine.groupBy("reject_reason").count().collect()}


def test_schema_drift_fails_fast(spark, raw_file):
    path = raw_file(["1/1/2007;00:00:00"], header="Date;Time")
    with pytest.raises(bronze_layer.SchemaDriftError, match="Voltage"):
        bronze_layer.read_raw(spark, path)


def test_each_bad_row_is_quarantined_with_its_reason(spark, raw_file):
    path = raw_file([
        reading(time="00:00:00"),
        "1/1/2007;00:01:00;?;?;?;?;?;?;?",
        reading(time="00:02:00", power="?"),
        reading(time="00:03:00", power="-1"),
        reading(time="00:04:00", voltage="999"),
        reading(date="not-a-date"),
    ])
    silver, quarantine = silver_from(spark, path)
    assert silver.count() == 1
    assert reasons(quarantine) == {
        "missing_reading": 1, "partial_reading": 1, "negative_measure": 1,
        "voltage_out_of_range": 1, "invalid_timestamp": 1,
    }


def test_duplicate_timestamp_keeps_one_row_and_quarantines_the_rest(spark, raw_file):
    silver, quarantine = silver_from(spark, raw_file([reading(), reading()]))
    assert silver.count() == 1
    assert reasons(quarantine) == {"duplicate_timestamp": 1}


def test_corrupt_copy_does_not_displace_good_reading(spark, raw_file):
    silver, _ = silver_from(spark, raw_file([reading(voltage="999"), reading()]))
    assert silver.first()["voltage_v"] == 240.0


def test_missing_hour_is_null_not_zero(spark, raw_file):
    path = raw_file([reading(time="00:00:00"), reading(time="02:00:00")])
    silver, _ = silver_from(spark, path)
    hourly = gold_layer.build_gold(spark, silver)["hourly_energy"].orderBy("hour_ts").collect()
    assert len(hourly) == 3
    assert hourly[1]["readings"] == 0
    assert hourly[1]["energy_kwh_observed"] is None
    assert hourly[1]["is_complete"] is False


def test_partial_hour_estimates_full_hour_from_average_power(spark, raw_file):
    path = raw_file([reading(time=f"00:{m:02d}:00", power="3.0") for m in range(30)])
    silver, _ = silver_from(spark, path)
    row = gold_layer.build_gold(spark, silver)["hourly_energy"].first()
    assert row["coverage"] == 0.5
    assert row["energy_kwh_observed"] == pytest.approx(1.5)
    assert row["energy_kwh_estimated"] == pytest.approx(3.0)


def test_gap_detection_reports_exact_missing_minutes(spark, raw_file):
    path = raw_file([reading(time="00:00:00"), reading(time="00:01:00"), reading(time="00:06:00")])
    silver, _ = silver_from(spark, path)
    gaps = gold_layer.build_gold(spark, silver)["data_gaps"].collect()
    assert [g["missing_minutes"] for g in gaps] == [4]


def _weekly_hours(spark, values):
    """Same weekday and hour across consecutive weeks, as complete hourly rows."""
    rows = [(i, float(v)) for i, v in enumerate(values)]
    df = spark.createDataFrame(rows, "week INT, energy DOUBLE")
    hourly = df.select(
        F.expr("timestamp'2007-01-01 12:00:00' + make_dt_interval(week * 7)").alias("hour_ts"),
        F.lit(True).alias("is_complete"),
        F.col("energy").alias("energy_kwh_observed"),
    )
    hourly.createOrReplaceTempView("gold_hourly_energy")
    return gold_layer.run_sql(spark, "gold_hourly_anomalies")


def test_anomaly_flags_drop_against_prior_weeks(spark):
    result = _weekly_hours(spark, [1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.1]).collect()
    assert len(result) == 1
    assert result[0]["direction"] == "drop"


def test_anomaly_baseline_never_sees_future_values(spark):
    normal = [1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.1]
    with_huge_future = [*normal, 500.0]
    first = _weekly_hours(spark, normal).filter("direction = 'drop'").first()
    second = _weekly_hours(spark, with_huge_future).filter("direction = 'drop'").first()
    assert first["baseline_kwh"] == second["baseline_kwh"]


def test_gate_fails_when_rows_go_missing(spark, raw_file):
    silver, quarantine = silver_from(spark, raw_file([reading()]))
    results = quality.check_silver(bronze_count=5, silver=silver, quarantine=quarantine)
    with pytest.raises(quality.QualityGateError, match="rows_unaccounted_for"):
        quality.enforce(results)


def test_gate_fails_when_too_much_is_quarantined(spark, raw_file):
    path = raw_file(["1/1/2007;00:01:00;?;?;?;?;?;?;?", reading()])
    silver, quarantine = silver_from(spark, path)
    results = quality.check_silver(2, silver, quarantine, max_quarantine_pct=5)
    with pytest.raises(quality.QualityGateError, match="quarantine_pct"):
        quality.enforce(results)


def test_end_to_end_run_is_repeatable_and_reconciles(spark, raw_file, tmp_path):
    path = raw_file([reading(time=f"00:{m:02d}:00") for m in range(60)])
    lake = str(tmp_path / "lake")
    first = run_pipeline(spark, path, lake)
    second = run_pipeline(spark, path, lake)
    assert first["row_counts"]["silver"] == second["row_counts"]["silver"] == 60
    assert first["row_counts"]["gold"] == second["row_counts"]["gold"]
    assert all(check["passed"] for check in first["checks"])
    json.dumps(first)


def test_backfill_replaces_only_the_years_in_the_new_file(spark, raw_file, tmp_path):
    target = str(tmp_path / "bronze")
    old = bronze_layer.read_raw(spark, raw_file([reading(date="1/1/2007"), reading(date="1/1/2008")]))
    bronze_layer.write_bronze(bronze_layer.to_bronze(old), target)
    late = bronze_layer.read_raw(spark, raw_file([reading(date="2/1/2008", power="9.0")]))
    bronze_layer.write_bronze(bronze_layer.to_bronze(late), target)

    rows = spark.read.parquet(target).collect()
    assert {(r["year"], r["Global_active_power"]) for r in rows} == {(2007, "1.2"), (2008, "9.0")}


def test_rebuild_removes_stale_silver_partitions(spark, raw_file, tmp_path):
    silver_path, quarantine_path = str(tmp_path / "s"), str(tmp_path / "q")
    first = silver_from(spark, raw_file([reading(date="1/1/2007"), reading(date="1/1/2008")]))
    silver_layer.write_silver(*first, silver_path, quarantine_path)
    second = silver_from(spark, raw_file([reading(date="1/1/2008")]))
    silver_layer.write_silver(*second, silver_path, quarantine_path)
    assert spark.read.parquet(silver_path).count() == 1


def test_partition_pruning_is_used_for_year_filter(spark, raw_file, tmp_path):
    path = raw_file([reading(date="1/1/2007"), reading(date="1/1/2008")])
    silver, quarantine = silver_from(spark, path)
    target = str(tmp_path / "silver")
    silver_layer.write_silver(silver, quarantine, target, str(tmp_path / "q"))
    plan = spark.read.parquet(target).filter("year = 2008")._jdf.queryExecution()
    assert "PartitionFilters: [isnotnull(year" in plan.executedPlan().toString()

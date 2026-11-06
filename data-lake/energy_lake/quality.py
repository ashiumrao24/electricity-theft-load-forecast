"""Quality gates between layers. A failed gate stops bad data reaching gold."""
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class QualityGateError(Exception):
    pass


@dataclass
class CheckResult:
    name: str
    value: float
    limit: str
    passed: bool


def check_silver(bronze_count: int, silver: DataFrame, quarantine: DataFrame,
                 max_quarantine_pct: float = 5.0) -> list[CheckResult]:
    silver_count = silver.count()
    quarantine_count = quarantine.count()
    unaccounted = bronze_count - silver_count - quarantine_count
    duplicate_keys = silver_count - silver.select("reading_ts").distinct().count()
    null_keys = silver.filter(F.col("reading_ts").isNull()).count()
    quarantine_pct = 100 * quarantine_count / bronze_count if bronze_count else 100.0
    return [
        CheckResult("rows_unaccounted_for", unaccounted, "= 0", unaccounted == 0),
        CheckResult("duplicate_timestamps_in_silver", duplicate_keys, "= 0", duplicate_keys == 0),
        CheckResult("null_timestamps_in_silver", null_keys, "= 0", null_keys == 0),
        CheckResult("quarantine_pct", round(quarantine_pct, 4), f"<= {max_quarantine_pct}",
                    quarantine_pct <= max_quarantine_pct),
    ]


def check_gold(silver: DataFrame, hourly: DataFrame) -> list[CheckResult]:
    """Energy must survive aggregation: the hourly total has to equal the minute-level total."""
    silver_kwh = silver.agg(F.sum("active_power_kw")).first()[0] / 60
    hourly_kwh = hourly.agg(F.sum("energy_kwh_observed")).first()[0]
    difference = abs(silver_kwh - hourly_kwh)
    duplicate_hours = hourly.count() - hourly.select("hour_ts").distinct().count()
    return [
        CheckResult("hourly_vs_minute_energy_diff_kwh", round(difference, 6), "< 0.001",
                    difference < 0.001),
        CheckResult("duplicate_hours_in_gold", duplicate_hours, "= 0", duplicate_hours == 0),
    ]


def enforce(results: list[CheckResult]) -> None:
    failed = [r.name for r in results if not r.passed]
    if failed:
        raise QualityGateError(f"Quality gate failed: {failed}")

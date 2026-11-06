"""Gold: business tables built with Spark SQL. The SQL lives in sql/*.sql."""
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def run_sql(spark: SparkSession, name: str) -> DataFrame:
    return spark.sql((SQL_DIR / f"{name}.sql").read_text(encoding="utf-8"))


def build_gold(spark: SparkSession, silver: DataFrame) -> dict[str, DataFrame]:
    silver.createOrReplaceTempView("silver_readings")
    hourly = run_sql(spark, "gold_hourly_energy").cache()
    hourly.createOrReplaceTempView("gold_hourly_energy")
    daily = run_sql(spark, "gold_daily_usage").cache()
    daily.createOrReplaceTempView("gold_daily_usage")
    return {
        "hourly_energy": hourly,
        "daily_usage": daily,
        "monthly_kpis": run_sql(spark, "gold_monthly_kpis"),
        "data_gaps": run_sql(spark, "gold_data_gaps"),
        "hourly_anomalies": run_sql(spark, "gold_hourly_anomalies"),
    }


def write_gold(tables: dict[str, DataFrame], gold_path: str) -> dict[str, int]:
    counts = {}
    for name, table in tables.items():
        writer = table.write.mode("overwrite").option("partitionOverwriteMode", "static")
        if "year" in table.columns:
            writer = writer.partitionBy("year")
        writer.parquet(f"{gold_path}/{name}")
        counts[name] = table.count()
    return counts

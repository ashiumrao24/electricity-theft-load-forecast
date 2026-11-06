"""Export the small gold tables as CSV for BI tools, and print headline facts."""
import csv
from pathlib import Path

from pyspark.sql import functions as F

from energy_lake.session import build_spark

GOLD = "data/lake/gold"


def export(frame, name: str) -> None:
    with Path(f"reports/{name}.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(frame.columns)
        writer.writerows(tuple(row) for row in frame.collect())


if __name__ == "__main__":
    spark = build_spark("export")
    spark.sparkContext.setLogLevel("ERROR")
    Path("reports").mkdir(exist_ok=True)
    export(spark.read.parquet(f"{GOLD}/monthly_kpis").orderBy("month_start"), "monthly_kpis")
    export(spark.read.parquet(f"{GOLD}/data_gaps").orderBy(F.desc("missing_minutes")), "data_gaps")

    spark.read.parquet(f"{GOLD}/hourly_energy").createOrReplaceTempView("hourly")
    spark.read.parquet(f"{GOLD}/hourly_anomalies").createOrReplaceTempView("anomalies")
    spark.read.parquet(f"{GOLD}/data_gaps").createOrReplaceTempView("gaps")
    spark.sql("""
        SELECT COUNT(*) AS hours,
               SUM(CASE WHEN readings = 0 THEN 1 ELSE 0 END) AS empty_hours,
               SUM(CASE WHEN readings > 0 AND NOT is_complete THEN 1 ELSE 0 END) AS partial_hours,
               SUM(CASE WHEN is_complete THEN 1 ELSE 0 END) AS complete_hours
        FROM hourly
    """).show()
    spark.sql("SELECT direction, COUNT(*) AS hours FROM anomalies GROUP BY direction").show()
    spark.sql("""
        SELECT COUNT(*) AS gaps, SUM(missing_minutes) AS missing_minutes,
               MAX(missing_minutes) AS longest_gap_minutes
        FROM gaps
    """).show()
    spark.stop()

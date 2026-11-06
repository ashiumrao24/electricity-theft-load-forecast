"""Compare the same one-year aggregation on the raw text file and on partitioned Parquet."""
import json
import statistics
import time
from pathlib import Path

from pyspark.sql import functions as F

from energy_lake.session import build_spark

RUNS = 5
RAW = "../data/household_power_consumption.txt"
SILVER = "data/lake/silver/readings"


def query_raw(spark):
    df = spark.read.option("header", True).option("sep", ";").csv(RAW)
    return (
        df.filter(F.col("Date").endswith("/2009"))
        .agg(F.sum(F.col("Global_active_power").cast("double")) / 60)
        .first()[0]
    )


def query_parquet(spark):
    df = spark.read.parquet(SILVER).filter("year = 2009")
    return df.agg(F.sum("active_power_kw") / 60).first()[0]


def timed(function, spark):
    function(spark)  # warm-up run, not measured
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        result = function(spark)
        samples.append(time.perf_counter() - started)
    return result, round(statistics.median(samples), 3)


if __name__ == "__main__":
    spark = build_spark("benchmark")
    spark.sparkContext.setLogLevel("ERROR")
    raw_result, raw_seconds = timed(query_raw, spark)
    parquet_result, parquet_seconds = timed(query_parquet, spark)
    spark.stop()
    report = {
        "query": "total 2009 energy in kWh",
        "runs": RUNS,
        "raw_text_median_seconds": raw_seconds,
        "parquet_median_seconds": parquet_seconds,
        "speedup": round(raw_seconds / parquet_seconds, 1),
        "raw_result_kwh": round(raw_result, 3),
        "parquet_result_kwh": round(parquet_result, 3),
    }
    Path("reports/benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

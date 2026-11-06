"""Silver: typed, deduplicated, validated readings. Failures go to quarantine with a reason."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

MEASURES = {
    "Global_active_power": "active_power_kw",
    "Global_reactive_power": "reactive_power_kw",
    "Voltage": "voltage_v",
    "Global_intensity": "current_a",
    "Sub_metering_1": "kitchen_wh",
    "Sub_metering_2": "laundry_wh",
    "Sub_metering_3": "water_heater_ac_wh",
}
MIN_VOLTAGE, MAX_VOLTAGE = 180.0, 280.0


def _typed(bronze: DataFrame) -> DataFrame:
    typed = bronze.withColumn(
        "reading_ts",
        F.to_timestamp(F.concat_ws(" ", "Date", "Time"), "d/M/yyyy HH:mm:ss"),
    )
    for source, target in MEASURES.items():
        # The source writes '?' for a missing value, and casting that to double gives null.
        typed = typed.withColumn(target, F.col(source).cast("double"))
    return typed


def _rejection_reason() -> F.Column:
    measures = list(MEASURES.values())
    all_null = F.lit(True)
    any_null = F.lit(False)
    for column in measures:
        all_null = all_null & F.col(column).isNull()
        any_null = any_null | F.col(column).isNull()
    negative = (
        (F.col("active_power_kw") < 0) | (F.col("reactive_power_kw") < 0)
        | (F.col("current_a") < 0) | (F.col("kitchen_wh") < 0)
        | (F.col("laundry_wh") < 0) | (F.col("water_heater_ac_wh") < 0)
    )
    return (
        F.when(F.col("reading_ts").isNull(), "invalid_timestamp")
        .when(all_null, "missing_reading")
        .when(any_null, "partial_reading")
        .when(negative, "negative_measure")
        .when(~F.col("voltage_v").between(MIN_VOLTAGE, MAX_VOLTAGE), "voltage_out_of_range")
    )


def to_silver(bronze: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return (clean readings, quarantined rows)."""
    checked = _typed(bronze).withColumn("reject_reason", _rejection_reason())

    # Only rows that passed validation compete for a timestamp, so a corrupt copy
    # can never displace a good reading.
    newest_first = Window.partitionBy("reading_ts").orderBy(F.col("_ingested_at").desc())
    valid = checked.filter(F.col("reject_reason").isNull()).withColumn(
        "copy_number", F.row_number().over(newest_first)
    )

    silver = (
        valid.filter(F.col("copy_number") == 1)
        .withColumn("year", F.year("reading_ts"))
        .withColumn("month", F.month("reading_ts"))
        .select("reading_ts", *MEASURES.values(), "_source_file", "year", "month")
    )
    duplicates = (
        valid.filter(F.col("copy_number") > 1)
        .withColumn("reject_reason", F.lit("duplicate_timestamp"))
        .drop("copy_number")
    )
    rejected = checked.filter(F.col("reject_reason").isNotNull())
    quarantine = rejected.unionByName(duplicates).select(
        "Date", "Time", *MEASURES.keys(), "reject_reason", "_source_file", "_ingested_at"
    )
    return silver, quarantine


def write_silver(silver: DataFrame, quarantine: DataFrame, silver_path: str,
                 quarantine_path: str) -> None:
    # Silver is rebuilt in full from bronze, so stale partitions must be removed.
    silver.write.mode("overwrite").option("partitionOverwriteMode", "static") \
        .partitionBy("year", "month").parquet(silver_path)
    quarantine.write.mode("overwrite").option("partitionOverwriteMode", "static") \
        .partitionBy("reject_reason").parquet(quarantine_path)

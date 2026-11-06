"""Bronze: land the raw file unchanged as Parquet, with lineage columns added."""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

RAW_COLUMNS = [
    "Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


class SchemaDriftError(Exception):
    """The raw file does not have the expected columns."""


def read_raw(spark: SparkSession, path: str) -> DataFrame:
    # Everything stays a string here: bronze records what arrived, not what we wish arrived.
    df = spark.read.option("header", True).option("sep", ";").csv(path)
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise SchemaDriftError(f"Raw file is missing columns: {missing}")
    return df.select(*RAW_COLUMNS)


def to_bronze(raw: DataFrame) -> DataFrame:
    return (
        raw.withColumn("_source_file", F.input_file_name())
        .withColumn("_ingested_at", F.current_timestamp())
        # The raw date is d/M/yyyy, so the year is the text after the last slash.
        .withColumn("year", F.substring_index("Date", "/", -1).cast("int"))
    )


def write_bronze(bronze: DataFrame, path: str) -> None:
    """Replace only the years in this file, so a late file for one year is a safe backfill."""
    bronze.write.mode("overwrite").partitionBy("year").parquet(path)

"""Create the Spark session used by the pipeline and the tests."""
import os
import sys

from pyspark.sql import SparkSession


def _ensure_java() -> None:
    """Use a system JAVA_HOME if set, otherwise the Java runtime bundled by jdk4py."""
    if not os.environ.get("JAVA_HOME"):
        import jdk4py

        os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)


def build_spark(app_name: str = "energy-data-lake", shuffle_partitions: int = 8) -> SparkSession:
    _ensure_java()
    # Workers must use the same interpreter as the driver, not whatever is first on PATH.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    return (
        SparkSession.builder.master(os.getenv("SPARK_MASTER", "local[*]"))
        .appName(app_name)
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Default for bronze: rewrite only the partitions present in the incoming data.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        # Meter timestamps have no zone. UTC avoids daylight-saving gaps in the hourly spine.
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

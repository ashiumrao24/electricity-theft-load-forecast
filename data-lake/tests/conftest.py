import pytest

from energy_lake.session import build_spark

HEADER = ("Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;"
          "Sub_metering_1;Sub_metering_2;Sub_metering_3")


def reading(date="1/1/2007", time="00:00:00", power="1.2", voltage="240.0"):
    return f"{date};{time};{power};0.1;{voltage};5.0;0.0;1.0;17.0"


@pytest.fixture(scope="session")
def spark():
    session = build_spark("tests", shuffle_partitions=2)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def raw_file(tmp_path):
    def write(lines, header=HEADER):
        path = tmp_path / "raw.txt"
        path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
        return str(path)

    return write

"""Download the UCI Individual Household Electric Power Consumption dataset."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
TARGET = Path("../data")

if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(TARGET)
    print(f"Saved to {TARGET / 'household_power_consumption.txt'}")

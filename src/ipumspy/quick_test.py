
import tempfile
import pytest
from pathlib import Path
from zipfile import ZipFile
import os

from ipumspy import fileutils

# def nhgis_fixtures_path() -> Path:
#     path = Path(__file__)
#     return path.absolute().parent / "tests" / "fixtures" / "nhgis"

nhgis_fixtures_path = Path("tests/fixtures/nhgis")

# fileutils.find_files_in(nhgis_fixtures_path / "nhgis0707_csv.zip", "c")
# fileutils.find_files_in(nhgis_fixtures_path / "nhgis0707_csv.zip", "csv")

file = fileutils.find_files_in(
        nhgis_fixtures_path / "nhgis0712_csv.zip",
        name_ext="csv",
        file_select="ds135",
        multiple_ok=False,
        none_ok=False
        )

fwf_dir = tempfile.TemporaryDirectory()

with ZipFile(nhgis_fixtures_path / "nhgis0712_csv.zip", 'r') as zip_ref:
    zip_ref.extractall(fwf_dir.name)

# Construct path to the extracted file
file_path = os.path.join(fwf_dir.name, file)

res = fileutils.find_files_in(nhgis_fixtures_path / "nhgis0712_csv.zip", name_ext="csv", file_select="ds135")

print(res)

res = fileutils.find_files_in(nhgis_fixtures_path / "nhgis0730_fixed.zip", name_ext="dat")

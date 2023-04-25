import tempfile
import pytest
from pathlib import Path
from zipfile import ZipFile
import os

from ipumspy import fileutils


def test_open_or_yield(capsys):
    # Test if you pass a filename
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with fileutils.open_or_yield(tmpdir / "test.txt", "wt") as outfile:
            outfile.write("hello")

        with open(tmpdir / "test.txt", "rt") as infile:
            assert infile.read() == "hello"

    # Test if you pass an open file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with open(tmpdir / "test.txt", "wt") as outfile:
            outfile.write("again")

        with open(tmpdir / "test.txt", "rt") as infile:
            with fileutils.open_or_yield(infile) as wrapped_infile:
                assert wrapped_infile.read() == "again"

    # Test if you pass '-'
    with fileutils.open_or_yield("-", "wt") as outfile:
        outfile.write("test me")

    captured = capsys.readouterr()
    assert captured.out == "test me"

    # Test if you pass None
    with fileutils.open_or_yield("-", "wt") as outfile:
        outfile.write("test me again")

    captured = capsys.readouterr()
    assert captured.out == "test me again"

def test_find_files_in(nhgis_fixtures_path: Path):
    """
    Test error conditions for find_files_in
    """

    # test is_zip, is_dir, and exists() methods
    assert fileutils.is_zip(nhgis_fixtures_path / "nhgis0707_csv.zip") == True
    assert fileutils.is_dir(nhgis_fixtures_path) == True

    # get a standalone file from fixture
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

    # test if file_path exists in os, after being extracted and put in tempdir
    assert fileutils.exists(file_path) == True

    # mismatched name_ext and file_path
    with pytest.raises(ValueError):
        # see above definition for file_path, which refers to a csv
        fileutils.find_files_in(file_path, name_ext="dat")

    # bad file extension
    with pytest.raises(ValueError):
        fileutils.find_files_in(nhgis_fixtures_path / "nhgis0707_csv.zip", "c", none_ok=False)

    # multiple files found
    with pytest.raises(ValueError):
        fileutils.find_files_in(nhgis_fixtures_path / "nhgis0712_csv.zip", "csv")

    # single file found, returns one str
    res = fileutils.find_files_in(nhgis_fixtures_path / "nhgis0712_csv.zip", name_ext="csv", file_select="ds135")
    assert res == "nhgis0712_csv/nhgis0712_ds135_1990_pmsa.csv"

    # find the codebook by changing name_ext to "txt" (useful for readers, when we want to retrieve codebook)
    res = fileutils.find_files_in(nhgis_fixtures_path / "nhgis0712_csv.zip", name_ext="txt", file_select="ds135")
    assert res == "nhgis0712_csv/nhgis0712_ds135_1990_pmsa_codebook.txt"

    # ensure that it can locate .dat and .do files as well, for nhgis_read_fwf()
    res = fileutils.find_files_in(nhgis_fixtures_path / "nhgis0730_fixed.zip", name_ext="dat", file_select="ts")
    assert res == "nhgis0730_fixed/nhgis0730_ts_nominal_state.dat"

    res = fileutils.find_files_in(nhgis_fixtures_path / "nhgis0730_fixed.zip", name_ext="do", file_select="ts")
    assert res == "nhgis0730_fixed/nhgis0730_ts_nominal_state.do"


    
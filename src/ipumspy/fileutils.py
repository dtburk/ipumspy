# This file is part of ipumspy.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/ipumspy

"""
Utilities for interacting with the IPUMS file format
"""
import gzip
import io
import sys
import zipfile
import os
import re
from contextlib import contextmanager
from pathlib import Path, PosixPath
from typing import ContextManager, Optional

from .types import FileType


@contextmanager
def xml_opener(ddi_file: FileType):
    """
    Yield an opened XML file with method 'rb'. Can be any of the following:
        * An already opened file (we just yield it back)
        * A path to an XML file or a gzipped XML file (we open and yield)
        * A path to a directory or ZIP file (we find an XML file therein,
            open and yield)
    Raises:
        OSError: If the passed path does not exist
        ValueError: If the path does not contain a *unique* XML file
    """
    if isinstance(ddi_file, io.IOBase):
        # If we have been passed an opened file, just yield it back
        yield ddi_file
        return

    if isinstance(ddi_file, str):
        # Always wrap paths in Paths
        ddi_file = Path(ddi_file)

    if isinstance(ddi_file, Path):
        if not ddi_file.exists():
            raise OSError(f"File {ddi_file} does not exist")

        if ddi_file.is_dir():
            # Search for a unique XML file in a directory
            possible_files = [
                filename for filename in ddi_file.iterdir() if filename.suffix == ".xml"
            ]
            if not possible_files:
                raise ValueError(f"{ddi_file} contains no XML files")
            if len(possible_files) > 1:
                raise ValueError(f"{ddi_file} contains more than one XML file")
            with open(possible_files[0], "rb") as infile:
                yield infile

        elif ddi_file.suffix == ".zip":
            # ZIP files are essentially directories, so perform the same check
            with zipfile.ZipFile(ddi_file) as inzip:
                possible_files = [
                    zipinfo
                    for zipinfo in inzip.infolist()
                    if zipinfo.filename.endswith(".xml")
                ]
                if not possible_files:
                    raise ValueError(f"{ddi_file} contains no XML files")
                if len(possible_files) > 1:
                    raise ValueError(f"{ddi_file} contains more than one XML file")
                data = inzip.read(possible_files[0])
                yield io.BytesIO(data)

        elif ddi_file.suffix == ".gz":
            # We assume that gzipped files are directly the XML
            if len(ddi_file.suffixes) < 2 or ddi_file.suffixes[-2] != ".xml":
                raise ValueError(f"{ddi_file} ends with .gz but not .xml.gz")
            with gzip.open(ddi_file, "rb") as infile:
                yield infile

        elif ddi_file.suffix != ".xml":
            # Otherwise, we require the file be a text file containing the XML
            raise ValueError(f"Invalid format for ddi_file: {ddi_file.suffix}")
        else:
            with open(ddi_file, "rb") as infile:
                yield infile


@contextmanager
def data_opener(data_file: FileType, encoding: str = "iso-8859-1", mode: str = "rt"):
    """
    Yield an opened data file. Can be any of the following:
        * An already opened file (we just yield it back)
        * A path to a dat file or a gzipped dat file (we open and yield)

    Args:
        data_file: The path as described above
        encoding: The encoding of the data file. ISO-8859-1 is the IPUMS default
        mode: The mode to open the file in

    Raises:
        OSError: If the passed path does not exist
        ValueError: If the path does not contain a *unique* XML file
    """
    if isinstance(data_file, io.IOBase):
        # If we have been passed an opened file, just yield it back
        yield data_file
        return

    data_file = Path(data_file)

    if not data_file.exists():
        # If it ends in .gz, try removing the .gz
        if data_file.suffix == ".gz":
            try_data_file = data_file.with_suffix("")
            if not try_data_file.exists():
                raise OSError(f"File {data_file} does not exist")
            data_file = try_data_file

        # See if *adding* .gz makes it exist
        else:
            try_data_file = data_file.with_suffix(data_file.suffix + ".gz")
            if not try_data_file.exists():
                raise OSError(f"File {data_file} does not exist")
            data_file = try_data_file

    # OK, so now we have an existent file! Let's try to open it
    if data_file.suffix == ".gz":
        with gzip.open(data_file, "rt", encoding=encoding) as infile:
            yield infile
    else:
        if "b" in mode:
            # Binary mode. Does not take encoding argument
            with open(data_file, mode=mode) as infile:
                yield infile
        else:
            with open(data_file, mode=mode, encoding=encoding) as infile:
                yield infile


@contextmanager
def open_or_yield(
    filename: Optional[FileType], mode: str = "rt"
) -> ContextManager[io.IOBase]:
    """
    Yield an opened data file with the passed mode. Can be any of the following:
        * An already opened file (we just yield it back, ignoring mode)
        * "-" in which case sys.stdout is yielded, ignoring mode
        * A path to file, which is then opened with the passed mode

    Args:
        filename: The name of the file to open
        mode: The mode in which to open the file

    Raises:
        OSError: If the passed path does not exist
        ValueError: If the path does not contain a *unique* XML file
    """
    if isinstance(filename, io.IOBase):
        yield filename
        return

    if (not filename) or (filename == "-"):
        yield sys.stdout
        return

    with open(filename, mode) as opened_file:
        yield opened_file

def find_files_in(
        filepath,
        name_ext = None,
        file_select = None,
        multiple_ok = False,
        none_ok = False
):
    """
    Finds a single file within a .zip, directory, or returns standalone. Assumes path is valid.

    Args:
        filepath: The name of the path to search for a file
        name_ext: The extension used to narrow file search, e.g. .csv or .dat
        file_select: A keyword used to narrow file search, e.g. "ds135"
        multiple_ok: Multiple files are allowed
        none_ok: No files are allowed
    
    Raises:
        FileExistsError: If the provided filepath does not exist.
        ValueError: If the provided filepath contains more than one fileor filepath itself is more than one item, i.e. the search is too broad.
    """

    
    # Check if it's a PosixPath object
    # if isinstance(filepath, PosixPath):
    #     # Get the absolute path
    #     filepath = filepath.resolve()

    if len([filepath]) != 1:
        raise ValueError(f"{filepath} contains more than one path, provide a single path for file parsing.")

    if is_zip(filepath):
        with zipfile.ZipFile(str(filepath), 'r') as f:
            file_names = [name for name in f.namelist()]

    elif is_dir(filepath):
        filepath = Path(filepath)
        file_names = [f.name for f in filepath.iterdir()]
    elif exists(filepath): # standalone file
        if not name_ext in filepath:
            if none_ok:
                file_names = ""
            else:
                raise ValueError(f"Error: Expected {filepath} to match extension .{name_ext}, but got {Path(filepath).suffix}")
        
        return Path(filepath)
    else:
        raise FileExistsError(f"Error: {filepath} does not exist in the user's OS.")


    if file_select is not None and name_ext is not None:
        matches = [s for s in file_names if re.findall(f".*{file_select}.*{name_ext}$", s)]
        file_names = matches
    elif name_ext is not None:
        matches = [s for s in file_names if re.findall(name_ext + "$", s)]
        file_names = matches
    elif file_select is not None:
        matches = [s for s in file_names if re.findall(file_select, s)]
        file_names = matches

    if not none_ok and len(file_names) == 0:
        raise ValueError(f"Did not find any files matching extension {name_ext} or matching the file_select {file_select} in the provided file path.")
    
    if not multiple_ok and len(file_names) > 1:
        
        raise ValueError(f"Multiple files found, please use the file_select and name_ext arguments to specify which file you want to load:\n{file_names}")

    else: # return the standalone file path, or the first element from the list of file_names (in a list with len == 1)
        return str(file_names) if type(file_names) != list else str(file_names[0])
    

def is_dir(file):
    """
    Returns boolean whether a given file is a directory
    """
    return os.path.isdir(file)

def is_zip(file):
    """
    Returns boolean whether a given file is a .zip archive
    """
    return zipfile.is_zipfile(file)

def exists(file):
    """
    Returns boolean whether a given file exists in the user's OS
    """
    return os.path.exists(file)
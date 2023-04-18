# This file is part of ipumspy.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/ipumspy

"""
Functions for reading and processing IPUMS data
"""
import copy
import json
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Optional, Union, Dict

import pandas as pd
import numpy as np
import yaml

from . import ddi as ddi_definitions
from . import fileutils
from .fileutils import open_or_yield
from .types import FilenameType


class CitationWarning(Warning):
    pass


def read_ipums_ddi(ddi_file: fileutils.FileType) -> ddi_definitions.Codebook:
    """
    Read a DDI from a IPUMS XML file

    Args:
        ddi_file: path to an IPUMS DDI XML

    Returns:
        A parsed IPUMS ddi codebook
    """
    with fileutils.xml_opener(ddi_file) as opened_file:
        root = ET.parse(opened_file).getroot()

    # Extract the namespace if there is one
    match = re.match(r"^\{(.*)\}", root.tag)
    namespace = match.groups()[0] if match else ""
    warnings.warn(
        "Use of data from IPUMS is subject to conditions including that users "
        "should cite the data appropriately.\n"
        "See the `ipums_conditions` attribute of this codebook for terms of use.\n"
        "See the `ipums_citation` attribute of this codebook for the appropriate "
        "citation.",
        CitationWarning,
    )
    return ddi_definitions.Codebook.read(root, namespace)


def _read_microdata(
    ddi: ddi_definitions.Codebook,
    filename: Optional[fileutils.FileType] = None,
    encoding: Optional[str] = None,
    subset: Optional[List[str]] = None,
    iterator: bool = False,
    chunksize: Optional[int] = None,
    dtype: Optional[dict] = None,
    **kwargs,
):
    # if ddi.file_description.structure != "rectangular":
    #     raise NotImplementedError("Structure must be rectangular")

    if subset is not None:
        data_description = [
            desc for desc in ddi.data_description if desc.name in subset
        ]
    else:
        data_description = ddi.data_description

    filename = Path(filename or ddi.file_description.filename)
    encoding = encoding or ddi.file_description.encoding

    iterator = iterator or (chunksize is not None)

    # Set up the correct reader for our file type
    kwargs = copy.deepcopy(kwargs)
    if ".dat" in filename.suffixes:
        # This is a fixed width file

        kwargs.update(
            {
                "colspecs": [(desc.start, desc.end) for desc in data_description],
                "names": [desc.name for desc in data_description],
                # numpy_type since _fix_decimal_expansion call will convert any shiftable integer columns to float anyway.
                "dtype": {desc.name: desc.numpy_type for desc in data_description},
            }
        )

        reader = pd.read_fwf
        mode = "rt"

        # Fixed width files also require fixing decimal expansions
        def _fix_decimal_expansion(df):
            for desc in data_description:
                if desc.shift:
                    shift = 10**desc.shift
                    df[desc.name] /= shift
            return df

    elif ".csv" in filename.suffixes:
        # A csv!
        reader = pd.read_csv
        kwargs.update(
            {
                "usecols": [desc.name for desc in data_description],
            }
        )

        if dtype is None:
            kwargs.update(
                {"dtype": {desc.name: desc.numpy_type for desc in data_description}}
            )
        else:
            kwargs.update({"dtype": dtype})

        mode = "rt"

        # CSVs have correct decimal expansions already; so we just make
        # this the identity function
        def _fix_decimal_expansion(df):
            return df

    elif ".parquet" in filename.suffixes:
        # A parquet file
        if dtype is not None:
            raise ValueError("dtype argument can't be used with parquet files.")
        reader = pd.read_parquet
        kwargs.update({"columns": [desc.name for desc in data_description]})
        mode = "rb"

        # Parquets have correct decimal expansions already; so we just make
        # this the identity function
        def _fix_decimal_expansion(df):
            return df

    else:
        raise ValueError("Only CSV and .dat files are supported")

    with fileutils.data_opener(filename, encoding=encoding, mode=mode) as infile:
        if not iterator:
            data = [reader(infile, **kwargs)]
        else:
            kwargs.update({"iterator": True, "chunksize": chunksize})
            data = reader(infile, **kwargs)

        if dtype is None:
            yield from (
                _fix_decimal_expansion(df).astype(
                    {desc.name: desc.pandas_type for desc in data_description}
                )
                for df in data
            )
        else:
            if ".dat" in filename.suffixes:
                # convert variables from default numpy_type to corresponding type in dtype.
                yield from (_fix_decimal_expansion(df).astype(dtype) for df in data)
            else:
                # In contrary to counter condition, df already has right dtype. It would be expensive to call astype for
                # nothing.
                yield from (_fix_decimal_expansion(df) for df in data)


def _read_hierarchical_microdata(
    ddi: ddi_definitions.Codebook,
    filename: Optional[fileutils.FileType] = None,
    encoding: Optional[str] = None,
    subset: Optional[List[str]] = None,
    iterator: bool = False,
    chunksize: Optional[int] = 100000,
    dtype: Optional[dict] = None,
    **kwargs,
):
    # TODO: try and speed this up
    if subset is not None:
        data_description = [
            desc for desc in ddi.data_description if desc.name in subset
        ]
    else:
        data_description = ddi.data_description

    # identify common variables
    # these variables have all rectypes listed in the variable-level rectype attribute
    # these are delimited by spaces within the string attribute
    # this list would probably be a useful thing to have as a file-level attribute...
    common_vars = [
        desc.name
        for desc in data_description
        if sorted(desc.rectype.split(" ")) == sorted(ddi.file_description.rectypes)
    ]
    # seperate variables by rectype
    rectypes = {}
    # NB: This might result in empty data frames for some rectypes
    # as the ddi contains all possible collection rectypes, even if only a few
    # are actually represented in the file.
    # TODO: prune empty rectype data frames
    for rectype in ddi.file_description.rectypes:
        rectype_vars = []
        rectype_vars.extend(common_vars)
        for desc in data_description:
            if desc.rectype == rectype:
                rectype_vars.append(desc.name)
        # read microdata for the relevant rectype variables only
        # and do it in chunks so it goes quicker
        rectypes[rectype] = next(
            read_microdata_chunked(
                ddi,
                filename,
                encoding,
                # rectype vars are the subset
                rectype_vars,
                chunksize,
                dtype,
                **kwargs,
            )
        )
        # retain only records from the relevant record type
        rt_df = rectypes[rectype]
        rectypes[rectype] = rt_df[rt_df["RECTYPE"] == rectype]
    return rectypes


def read_microdata(
    ddi: ddi_definitions.Codebook,
    filename: Optional[fileutils.FileType] = None,
    encoding: Optional[str] = None,
    subset: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    **kwargs,
) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
    """
    Read in microdata as specified by the Codebook. Both .dat and .csv file types
    are supported.

    Args:
        ddi: The codebook representing the data
        filename: The path to the data file. If not present, gets from
                        ddi and assumes the file is relative to the current
                        working directory
        encoding: The encoding of the data file. If not present, reads from ddi
        subset: A list of variable names to keep. If None, will keep all
        dtype: A dictionary with variable names as keys and variable types as values.
            Has an effect only when used with pd.read_fwf or pd.read_csv engine. If None, pd.read_fwf or pd.read_csv use
            type ddi.data_description.pandas_type for all variables. See ipumspy.ddi.VariableDescription for more
            precision on ddi.data_description.pandas_type. If files are csv, and dtype is not None, pandas converts the
            column types once: on pd.read_csv call. When file format is .dat or .csv and dtype is None, two conversion
            occur: one on load, and one when returning the dataframe.
        kwargs: keyword args to be passed to the engine (pd.read_fwf, pd.read_csv, or
            pd.read_parquet depending on the file type)

    Returns:
        pandas data frame and pandas text file reader
    """
    # raise a warning if this is a hierarchical file
    if ddi.file_description.structure == "hierarchical":
        raise NotImplementedError(
            "Structure must be rectangular. Use `read_hierarchical_microdata()` for hierarchical extracts."
        )
    # just read it if its rectangular
    else:
        return next(
            _read_microdata(
                ddi,
                filename=filename,
                encoding=encoding,
                subset=subset,
                dtype=dtype,
                **kwargs,
            )
        )


def read_hierarchical_microdata(
    ddi: ddi_definitions.Codebook,
    filename: Optional[fileutils.FileType] = None,
    encoding: Optional[str] = None,
    subset: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    as_dict: Optional[bool] = True,
    **kwargs,
) -> Union[pd.DataFrame, Dict]:
    """
    Read in microdata as specified by the Codebook. Both .dat and .csv file types
    are supported.

    Args:
        ddi: The codebook representing the data
        filename: The path to the data file. If not present, gets from
                        ddi and assumes the file is relative to the current
                        working directory
        encoding: The encoding of the data file. If not present, reads from ddi
        subset: A list of variable names to keep. If None, will keep all
        dtype: A dictionary with variable names as keys and variable types as values.
            Has an effect only when used with pd.read_fwf or pd.read_csv engine. If None, pd.read_fwf or pd.read_csv use
            type ddi.data_description.pandas_type for all variables. See ipumspy.ddi.VariableDescription for more
            precision on ddi.data_description.pandas_type. If files are csv, and dtype is not None, pandas converts the
            column types once: on pd.read_csv call. When file format is .dat or .csv and dtype is None, two conversion
            occur: one on load, and one when returning the dataframe.
        as_dict: A flag to indicate whether to return a single data frame or a dictionary with one data frame per record
            type in the extract data file. Set to True to recieve a dictionary of data frames.
        kwargs: keyword args to be passed to the engine (pd.read_fwf, pd.read_csv, or
            pd.read_parquet depending on the file type)

    Returns:
        pandas data frame or a dictionary of pandas data frames
    """
    # hack for now just to have it in this method - make this a ddi.file_description attribute.
    common_vars = [
        desc.name
        for desc in ddi.data_description
        if sorted(desc.rectype.split(" ")) == sorted(ddi.file_description.rectypes)
    ]
    # RECTYPE must be included if subset list is specified
    if subset is not None and "RECTYPE" not in subset:
        raise ValueError(
            "RECTYPE must be included in the subset list for hierarchical extracts."
        )
    # raise a warning if this is a rectantgular file
    if ddi.file_description.structure == "rectangular":
        raise NotImplementedError(
            "Structure must be hierarchical. Use `read_microdata()` for rectangular extracts."
        )
    else:
        df_dict = _read_hierarchical_microdata(
            ddi, filename, encoding, subset, dtype, **kwargs
        )
        if as_dict:
            return df_dict
        else:
            # read the hierarchical file
            df = next(_read_microdata(ddi, filename, encoding, subset, dtype, **kwargs))
            # for each rectype, nullify variables that belong to other rectypes
            for rectype in df_dict.keys():
                # create a list of variables that are for rectypes other than the current rectype
                # and are not included in the list of varaibles that are common across rectypes
                non_rt_cols = [
                    cols
                    for rt in df_dict.keys()
                    for cols in df_dict[rt].columns
                    if rt != rectype and cols not in common_vars
                ]
                for col in non_rt_cols:
                    # maintain data type when "nullifying" variables from other record types
                    if df[col].dtype == pd.Int64Dtype():
                        df[col] = np.where(df["RECTYPE"] == rectype, pd.NA, df[col])
                        df[col] = df[col].astype(pd.Int64Dtype())
                    elif df[col].dtype == pd.StringDtype():
                        df[col] = np.where(df["RECTYPE"] == rectype, "", df[col])
                        df[col] = df[col].astype(pd.StringDtype())
                    elif df[col].dtype == float:
                        df[col] = np.where(df["RECTYPE"] == rectype, np.nan, df[col])
                        df[col] = df[col].astype(float)
                    # this should (theoretically) never be hit... unless someone specifies an illegal data type
                    # themselves, but that should also be caught before this stage.
                    else:
                        raise TypeError(
                            f"Data type {df[col].dtype} for {col} is not an allowed type."
                        )
            return df


def read_microdata_chunked(
    ddi: ddi_definitions.Codebook,
    filename: Optional[fileutils.FileType] = None,
    encoding: Optional[str] = None,
    subset: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    dtype: Optional[dict] = None,
    **kwargs,
) -> Iterator[pd.DataFrame]:
    """
    Read in microdata in chunks as specified by the Codebook.
    As these files are often large, you may wish to filter or read in chunks.
    As an example of how you might do that, consider the following example that
    filters only for rows in Rhode Island::

        iter_microdata = read_microdata_chunked(ddi, chunksize=1000)
        df = pd.concat([df[df['STATEFIP'] == 44]] for df in iter_microdata])

    Args:
        ddi: The codebook representing the data
        filename: The path to the data file. If not present, gets from
                     ddi and assumes the file is relative to the current working directory
        encoding: The encoding of the data file. If not present, reads from ddi
        subset: A list of variable names to keep. If None, will keep all
        dtype: A dictionary with variable names as keys and variable types as values.
            Has an effect only when used with pd.read_fwf or pd.read_csv engine. If None, pd.read_fwf or pd.read_csv use
            type ddi.data_description.pandas_type for all variables. See ipumspy.ddi.VariableDescription for more
            precision on ddi.data_description.pandas_type. If files are csv, and dtype is not None, pandas converts the
            column types once: on pd.read_csv call. When file format is .dat or .csv and dtype is None, two conversion
            occur: one on load, and one when returning the dataframe.
        chunksize: The size of the chunk to return with iterator. See `pandas.read_csv`
        kwargs: keyword args to be passed to pd.read_fwf
    Yields:
        An iterator of data frames
    """
    yield from _read_microdata(
        ddi,
        filename=filename,
        encoding=encoding,
        subset=subset,
        iterator=True,
        dtype=dtype,
        chunksize=chunksize,
        **kwargs,
    )


def read_extract_description(extract_filename: FilenameType) -> dict:
    """
    Open an extract description (either yaml or json are accepted) and return it
    as a dictionary.

    Args:
        extract_filename: The path to the extract description file

    Returns:
        The contents of the extract description
    """
    with open_or_yield(extract_filename) as infile:
        data = infile.read()

    try:
        return json.loads(data)
    except json.decoder.JSONDecodeError:
        # Assume this is a yaml file and not a json file
        pass

    try:
        return yaml.safe_load(data)
    except yaml.error.YAMLError:
        raise ValueError("Contents of extract file appear to be neither json nor yaml")

def read_nhgis_fwf(data_file,
                   file_select = None,
                   do_file = None,
                   verbose = True,
                   *kwargs
                ):
    
    file = find_files_in(
        data_file,
        name_ext="dat",
        file_select=file_select,
        multiple_ok=False,
        none_ok=False
    )

    cb_files = find_files_in(
        data_file,
        name_ext="txt",
        multiple_ok=True,
        none_ok=True
    )

    if verbose:
        
        print("Codebook to be implemented.")

    if is_zip(data_file):
    # Cannot use fixed width format on a ZIP file
    # Must extract ZIP contents to allow for default format specification
        fwf_dir = tempfile.TemporaryDirectory()

        with ZipFile(data_file, 'r') as zip_ref:
            zip_ref.extractall(fwf_dir.name)

        # Construct path to the extracted file
        file_path = os.path.join(fwf_dir.name, file)

    elif is_dir(data_file):
        # Construct path to the file within the directory
        file_path = os.path.join(data_file, file)
    
    do_file = (file_path.rstrip(".dat") + ".do") or do_file # removes .dat and replaces with .do

    if do_file is None:
        warn_default_fwf_parsing()
    elif not exists_in_os(do_file):
        if do_file is not None:
            print(f"Could not find the provided do_file, {do_file}. \
                             Make sure the provided do_file exists or use 'col_positions' to specify \
                             column positions manually.")
            warn_default_fwf_parsing()
        else:
            print(f"Could not find a .do file associated with the provided file. \
                  Use the 'do_file' argument to provide an associated .do file \
                  or use 'col_positions' to specify column positions manually.")
            warn_default_fwf_parsing()
    elif exists_in_os(do_file):
        
        colspecs, names, dtype, replace_list = parse_nhgis_do_file(do_file)

        df = pd.read_fwf(file_path, colspecs=colspecs, names=names, dtype=dtype)

        for column in replace_list:
            # adjust for implicit decimal
            df[column] = df[column] / 10
        
        return df


def warn_default_fwf_parsing():

    print("Data loaded from NHGIS fixed-width files may not be consistent with " \
        "the information included in the data codebook when parsing column " \
        "positions manually.")

def parse_nhgis_do_file(file):
    """
        Extract relevant information from .do file in order to parse .dat file for
        read_nhgis_fwf(). Specifically, extracts column names, positions, dtypes,
        and takes note of implicit decimals.

        Args:
            file: The path to a .do file, used to parse a similarly-named .dat file
            (if name has been unchanged since download).

        Raises:
            OSError: If the passed path does not exist

        Returns:
            colspecs, names, dtype, and replace_list. These are specifications
            for column widths, column names, and data type by column (passed directly)
            into pd.read_fwf(), as well as list of columns that need to be "replaced"
            with their implicit decimal value (i.e. / 10)
        """
    
    # initialization of variables to be returned
    colspecs = []
    names = []
    dtype = {}
    replace_list = []

    # data type conversions, in dictionary form
    # Stata data type : Python data type
    # Note: Stata supports many types of numerics, Python does not
    var_type_conversion = {"int": "int", "byte": "float", "long": "float", "float": "float", "double": "float", "str": "str"}

    # open file
    do = open(file, "r")

    # booleans to identify what section of the .do file we're in
    spec_section = False
    replace_section = False

    # parse file
    for line in do:

        # "quietly infix" precedes column specification section
        if "quietly infix" in line:
            spec_section = True
            continue # move to next line, start parsing column specs

        # "replace" is followed by the first column with an implicit decimal
        if "replace" in line:
            replace_section = True
            # don't continue, we want to parse this line since it contains important info
        
        if spec_section:
            # "using" marks the end of the spec_section
            if "using" in line:
                spec_section = False
                continue

            # specs = [stata_var_type, var_name, col_width]
            specs = line.rstrip("///\n").split()

            VAR_TYPE = var_type_conversion[specs[0]]
            VAR_NAME = specs[1].upper()
            col_width = specs[2].split("-") # splits upper and lower bound into separate strings

            dtype[VAR_NAME] = VAR_TYPE # key is the variable name, value is the Python type of that variable
            names.append(VAR_NAME)
            colspecs.append(((int(col_width[0]) - 1), int(col_width[1]))) # give upper and lower bound of column width

        if replace_section:

            # if line is empty, no more variables to replace
            if line.isspace():
                replace_section = False
                break # exit the loop, nothing more to parse

            # get the name of the variable to be replaced from that line;
            # assumes that the variable being replaced has one implicit decimal, i.e.
            # that the variable must be divided by 10 across all observations.
            # This assumption is seen as safe for NHGIS data.
            var_name = line.strip().split("=")[0].rstrip().split(" ")[1].upper()

            replace_list.append(var_name)
            
        
    return colspecs, names, dtype, replace_list


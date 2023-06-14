"""
Wrappers for payloads to ship to the IPUMS API
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Type, Union

import requests
import json
import inspect

from ipumspy.ddi import Codebook

from dataclasses import dataclass, field
from .exceptions import IpumsExtractNotSubmitted

step = 0 # remove after debugging done
class DefaultCollectionWarning(Warning):
    pass


class ApiVersionWarning(Warning):
    pass


class ModifiedExtractWarning(Warning):
    pass


@dataclass
class Variable:
    """
    IPUMS variable object to include in an IPUMS extract object.
    """

    name: str
    """IPUMS variable name"""
    preselected: Optional[bool] = False
    """Whether the variable is preselected. Defaults to False."""
    case_selections: Optional[Dict[str, List]] = field(default_factory=dict)
    """Case selection specifications."""
    attached_characteristics: Optional[List[str]] = field(default_factory=list)
    """Attach characteristics specifications."""
    data_quality_flags: Optional[bool] = False
    """Flag to include the variable's associated data quality flags if they exist."""

    def __post_init__(self):
        self.name = self.name.upper()

    def update(self, attribute: str, value: Any):
        """
        Update Variable features

        Args:
            attribute: name of the Variable attribute to update
            value: values with which to update the `attribute`
        """
        if hasattr(self, attribute):
            setattr(self, attribute, value)
        else:
            raise KeyError(f"Variable has no attribute '{attribute}'.")

    def build(self):
        """Format Variable information for API Extract submission"""
        built_var = self.__dict__.copy()
        # don't repeat the variable name
        built_var.pop("name")
        # adhere to API schema camelCase convention
        built_var["caseSelections"] = built_var.pop("case_selections")
        built_var["attachedCharacteristics"] = built_var.pop("attached_characteristics")
        built_var["dataQualityFlags"] = built_var.pop("data_quality_flags")
        return built_var


@dataclass
class Sample:
    """
    IPUMS sample object to include in an IPUMS extract object.
    """

    id: str
    """IPUMS sample id"""
    description: Optional[str] = ""
    """IPUMS sample description"""

    def __post_init__(self):
        self.id = self.id.lower()

    def update(self, attribute: str, value: Any):
        """
        Update Sample features

        Args:
            attribute: name of the Sample attribute to update
            value: values with which to update the `attribute`
        """
        if hasattr(self, attribute):
            setattr(self, attribute, value)
        else:
            raise KeyError(f"Sample has no attribute '{attribute}'.")

@dataclass
class Dataset:
    """
    IPUMS dataset object to include in an IPUMS NHGIS extract object.
    """

    name: str
    """IPUMS dataset name"""
    data_tables: List[str]
    """IPUMS dataset datatables: Required"""
    geog_levels: List[str]
    """IPUMS dataset geog_levels: Required"""
    years: Optional[List[str]] = field(default_factory=list)
    """IPUMS dataset years"""
    breakdown_values: Optional[List[str]] = field(default_factory=list)
    """IPUMS dataset breakdown_values"""

    # field() is a factory function which provides a default value for the field, in this case an empty list.

    def __post_init__(self):
        self.name = self.name.upper()

    def update(self, attribute: str, value: Any):
        """
        Update Dataset features

        Args:
            attribute: name of the Dataset attribute to update
            value: values with which to update the `attribute`
        """
        if hasattr(self, attribute):
            setattr(self, attribute, value)
        else:
            raise KeyError(f"Dataset has no attribute '{attribute}'.")
        
    def build(self):

        built_dataset = self.__dict__.copy()
        # don't repeat the dataset name
        built_dataset.pop("name")
        # adhere to API schema camelCase convention
        built_dataset["dataTables"] = built_dataset.pop("data_tables")
        built_dataset["geogLevels"] = built_dataset.pop("geog_levels")
        built_dataset["years"] = built_dataset.pop("years")
        built_dataset["breakdownValues"] = built_dataset.pop("breakdown_values")

        return built_dataset

    def __str__(self):
        """
        For testing, print the dataset as a JSON string
        """
        return json.dumps(self.build())
    
@dataclass
class TimeSeriesTable:
    """
    IPUMS TimeSeriesTable object to include in an IPUMS NHGIS extract object.
    """
    
    name: str
    """IPUMS TimeSeriesTable name"""
    geog_levels: List[str] # required parameter
    """IPUMS TimeSeriesTable geog_levels: Required"""
    years: Optional[List[str]] = field(default_factory=list)
    """IPUMS TimeSeriesTable years"""

    # field() is a factory function which provides a default value for the field, in this case an empty list.

    def __post_init__(self):
        self.name = self.name.upper()

    def update(self, attribute: str, value: Any):
        """
        Update TimeSeriesTable features

        Args:
            attribute: name of the TimeSeriesTable attribute to update
            value: values with which to update the `attribute`
        """
        if hasattr(self, attribute):
            setattr(self, attribute, value)
        else:
            raise KeyError(f"TimeSeriesTable has no attribute '{attribute}'.")
        
    def build(self):

        built_time_series_table = self.__dict__.copy()
        # don't repeat the time series table name
        built_time_series_table.pop("name")
        # adhere to API schema camelCase convention
        built_time_series_table["geogLevels"] = built_time_series_table.pop("geog_levels")
        built_time_series_table["years"] = built_time_series_table.pop("years")

        return built_time_series_table
    
    def __str__(self):
        """
        For testing, print the dataset as a JSON string
        """
        return json.dumps(self.build())

def present_in_extract_info(item) -> bool:
        """
        Quick helper function that checks if an item is present in the extract info
        Makes code for build() more readable
        """
        if item is not None and len(item) > 0:
            return True

def _unpack_samples_dict(dct: dict) -> List[Sample]:
    return [Sample(id=samp) for samp in dct.keys()]


def _unpack_variables_dict(dct: dict) -> List[Variable]:
    vars = []
    for var in dct.keys():
        var_obj = Variable(name=var)
        # this feels dumb, but the best way to avoid KeyErrors
        # that is coming to my brain at the moment
        if "preselected" in dct[var]:
            var_obj.update("preselected", dct[var]["preselected"])
        if "caseSelections" in dct[var]:
            var_obj.update("case_selections", dct[var]["caseSelections"])
        if "attachedCharacteristics" in dct[var]:
            var_obj.update(
                "attached_characteristics", dct[var]["attachedCharacteristics"]
            )
        if "dataQualityFlags" in dct[var]:
            var_obj.update("data_quality_flags", dct[var]["dataQualityFlags"])
        vars.append(var_obj)
    return vars

def _unpack_datasets_dict(dct: dict) -> List[Dataset]:
    """"
    Following the example of _unpack_variables_dict, this function is intended to unpack the datasets dictionary
    into a list of Dataset objects. This behavior is needed for get_extract_by_id for nhgis extracts. This
    implementation is not complete and will need to be checked against API schema.
    """
    datasets = []
    for dataset in dct.keys():
        dataset_obj = Dataset(name=dataset, data_tables=dct[dataset]["dataTables"], geog_levels=dct[dataset]["geogLevels"])
        if "years" in dct[dataset]:
            dataset_obj.update("years", dct[dataset]["years"])
        if "breakdownValues" in dct[dataset]:
            dataset_obj.update("breakdown_values", dct[dataset]["breakdownValues"])
        datasets.append(dataset_obj)
    return datasets

def _unpack_time_series_tables_dict(dct: dict) -> List[TimeSeriesTable]:
    """"
    Following the example of _unpack_variables_dict, this function is intended to unpack the time series tables dictionary
    into a list of TimeSeriesTable objects. This behavior is needed for get_extract_by_id for nhgis extracts. This
    implementation is not complete and will need to be checked against API schema.
    """
    time_series_tables = []
    for time_series_table in dct.keys():
        time_series_table_obj = TimeSeriesTable(name=time_series_table, geog_levels=dct[time_series_table]["geogLevels"])
        if "years" in dct[time_series_table]:
            time_series_table_obj.update("years", dct[time_series_table]["years"])
        time_series_tables.append(time_series_table_obj)
    
    return time_series_tables

class BaseExtract:
    _collection_to_extract: Dict[(str, str), Type[BaseExtract]] = {}

    def __init__(self):
        """
        A wrapper around an IPUMS extract. In most cases, you
        probably want to use a subclass, but if a particular collection
        does not have an ``Extract`` currently built, you can use
        this wrapper directly.
        """

        self._id: Optional[int] = None
        self._info: Optional[Dict[str, Any]] = None
        self.api_version: Optional[str] = None

    def __init_subclass__(cls, collection: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.collection = collection
        BaseExtract._collection_to_extract[collection] = cls

    def _kwarg_warning(self, kwargs_dict: Dict[str, Any]):
        try:
            if kwargs_dict["collection"] == self.collection:
                # collection kwarg is same as default, nothing to do
                pass
            elif kwargs_dict["collection"] != self.collection:
                warnings.warn(
                    f"This extract object already has a default collection "
                    f"{self.collection}. Collection Key Word Arguments "
                    f"are ignored.",
                    DefaultCollectionWarning,
                )
        except KeyError:
            # if there collection isn't specified
            # then nothing to warn about there
            pass
        # raise kwarg warnings if they exist
        if "warnings" in kwargs_dict.keys():
            newline = "\n"
            warnings.warn(
                f"This extract object has been modified from its original form in the following ways: "
                f"{newline.join(kwargs_dict['warnings'])}",
                ModifiedExtractWarning,
            )

    def build(self) -> Dict[str, Any]:
        """
        Convert the object into a dictionary to be passed to the IPUMS API
        as a JSON string
        """
        raise NotImplementedError()

    @property
    def extract_id(self) -> int:
        """
        str:The extract id associated with this extract, assigned by the ``IpumsApiClient``

        Raises ``ValueError`` if the extract has no id number (probably because it has
        not be submitted to IPUMS)
        """
        if not self._id:
            raise ValueError("Extract has not been submitted so has no id number")
        return self._id

    @property
    def extract_info(self) -> Dict[str, Any]:
        """
        str: The API response recieved by the ``IpumsApiClient``

        Raises ``ValueError`` if the extract has no json response (probably because it
        has not bee submitted to IPUMS)
        """
        if not self._info:
            raise IpumsExtractNotSubmitted(
                "Extract has not been submitted and so has no json response"
            )
        else:
            return self._info

    def _snake_to_camel(self, kwarg_dict):
        for key in list(kwarg_dict.keys()):
            # create camelCase equivalent
            key_list = key.split("_")
            # join capitalized versions of all parts except the first
            camelized = "".join([k.capitalize() for k in key_list[1:]])
            # prepend the first part
            camel_key = f"{key_list[0]}{camelized}"
            # add the camelCase key
            kwarg_dict[camel_key] = kwarg_dict[key]
            # pop the snake_case key
            kwarg_dict.pop(key)

        return kwarg_dict

    def _validate_list_args(self, list_arg, arg_obj):
        # this bit feels extra sketch, but it seems like a better solution
        # than just having the BaseExtract(**kwargs) method of instantiating
        # an extract object quietly leave out variable-level extract features
        if isinstance(list_arg, dict) and arg_obj is Variable:
            args = _unpack_variables_dict(list_arg)
            return args
        elif isinstance(list_arg, dict) and arg_obj is Sample:
            args = _unpack_samples_dict(list_arg)
            return args
        elif isinstance(list_arg, dict) and arg_obj is Dataset:
            args = _unpack_datasets_dict(list_arg)
            return args
        elif isinstance(list_arg, dict) and arg_obj is TimeSeriesTable:
            args = _unpack_time_series_tables_dict(list_arg)
            return args
        # Make sure extracts don't get built with duplicate variables or samples
        # if the argument is a list of objects, make sure there are not objects with duplicate names
        elif all(isinstance(i, arg_obj) for i in list_arg):
            try:
                if len(set([i.name for i in list_arg])) < len(list_arg):
                    # Because Variable objects can have the same name but differet feature specifications
                    # force the user to fix this themselves
                    raise ValueError(
                        f"Duplicate Variable objects are not allowed in IPUMS Extract definitions."
                    )
                else:
                    # return the list of objects
                    return list_arg
            except AttributeError:
                if len(set([i.id for i in list_arg])) < len(list_arg):
                    # Because Sample objects can have the same id but differet feature specifications
                    # force the user to fix this themselves
                    raise ValueError(
                        f"Duplicate Sample objects are not allowed in IPUMS Extract definitions."
                    )
                else:
                    # return the list of objects
                    return list_arg
        elif all(isinstance(i, str) for i in list_arg):
            # if duplicate string names are specified, just drop the duplicates
            # and return a list of the relevant objects
            unique_list = list(dict.fromkeys(list_arg))
            return [arg_obj(i) for i in unique_list]

    def extract_api_version(self, kwargs_dict: Dict[str, Any]) -> str:
        # check to see if version is specified in kwargs_dict
        if "version" in kwargs_dict.keys() or "api_version" in kwargs_dict.keys():
            try:
                if kwargs_dict["version"] == self.api_version:
                    # collectin kwarg is the same as default, nothing to do
                    return self.api_version
                # this will only get hit if the extract object has already been submitted
                # or if an api_version other than None was explicitly passed to BaseExtract
                elif (
                    kwargs_dict["version"] != self.api_version
                    and self.api_version is not None
                ):
                    warnings.warn(
                        f"The IPUMS API version specified in the extract definition is not the most recent. "
                        f"Extract definition IPUMS API version: {kwargs_dict['version']}; most recent IPUMS API version: {self.api_version}",
                        ApiVersionWarning,
                    )
                    # update extract object api version to reflect
                    return kwargs_dict["version"]
                # In all other instances, return the version from the kwargs dict
                # If this version is illegal, it will raise an IpumsAPIAuthenticationError upon submission
                else:
                    return kwargs_dict["version"]
            except KeyError:
                # no longer supporting beta extract schema
                raise NotImplementedError(
                    f"The IPUMS API version specified in the extract definition is not supported by this version of ipumspy."
                )
        # if no api_version is specified, use default IpumsApiClient version
        else:
            return self.api_version

    def _update_variable_feature(self, variable, feature, specification):
        if isinstance(variable, Variable):
            variable.update(feature, specification)
        elif isinstance(variable, str):
            for var in self.variables:
                if var.name == variable:
                    var.update(feature, specification)
                    break
            else:
                raise ValueError(f"{variable} is not part of this extract.")
        else:
            raise TypeError(
                f"Expected a string or Variable object; {type(variable)} received."
            )

    def attach_characteristics(self, variable: Union[Variable, str], of: List[str]):
        """
        A method to update existing IPUMS Extract Variable objects
        with the IPUMS attach characteristics feature.

        Args:
            variable: a Variable object or a string variable name
            of: a list of records for which to create a variable on the individual record.
                Allowable values include "mother", "father", "spouse", "head". For IPUMS
                collection that identify same sex couples can also accept "mother2" and "father2"
                values in this list. If either "<parent>" or "<parent>2" values are included,
                their same sex counterpart will automatically be included in the extract.
        """
        self._update_variable_feature(variable, "attached_characteristics", of)

    def add_data_quality_flags(
        self, variable: Union[Variable, str, List[Variable], List[str]]
    ):
        """
        A method to update existing IPUMS Extract Variable objects to include that
        variable's data quality flag in the extract if it exists.

        Args:
            variable: a Variable object or a string variable name

        """
        if isinstance(variable, list):
            for v in variable:
                self._update_variable_feature(v, "data_quality_flags", True)
        else:
            self._update_variable_feature(variable, "data_quality_flags", True)

    def select_cases(
        self,
        variable: Union[Variable, str],
        values: List[Union[int, str]],
        general: bool = True,
    ):
        """
        A method to update existing IPUMS Extract Variable objects to select cases
        with the specified values of that IPUMS variable.

        Args:
            variable: a Variable object or a string variable name
            values: a list of values for which to select records
            general: set to False to select cases on detailed codes. Defaults to True.
        """
        # stringify values
        values = [str(v) for v in values]
        if general:
            self._update_variable_feature(
                variable, "case_selections", {"general": values}
            )
        else:
            self._update_variable_feature(
                variable, "case_selections", {"detailed": values}
            )


class OtherExtract(BaseExtract, collection="other"):
    def __init__(self, collection: str, details: Optional[Dict[str, Any]]):
        """
        A generic extract object for working with collections that are not
        yet officially supported by this API library
        """

        super().__init__()
        self.collection = collection
        """Name of an IPUMS data collection"""
        self.details = details
        """dictionary containing variable names and sample IDs"""

    def build(self) -> Dict[str, Any]:
        """
        Convert the object into a dictionary to be passed to the IPUMS API
        as a JSON string
        """
        return self.details


class UsaExtract(BaseExtract, collection="usa"):
    def __init__(
        self,
        samples: Union[List[str], List[Sample]],
        variables: Union[List[str], List[Variable]],
        description: str = "My IPUMS USA extract",
        data_format: str = "fixed_width",
        data_structure: Dict = {"rectangular": {"on": "P"}},
        **kwargs,
    ):
        """
        Defining an IPUMS USA extract.

        Args:
            samples: list of IPUMS USA sample IDs
            variables: list of IPUMS USA variable names
            description: short description of your extract
            data_format: fixed_width and csv supported
            data_structure: nested dict with "rectangular" or "hierarchical" as first-level key.
                            "rectangular" extracts require further specification of "on" : <record type>.
                            Default {"rectangular": "on": "P"} requests an extract rectangularized on the "P" record.
        """

        super().__init__()
        self.samples = self._validate_list_args(samples, Sample)
        self.variables = self._validate_list_args(variables, Variable)
        self.description = description
        self.data_format = data_format
        self.data_structure = data_structure
        self.collection = self.collection
        """Name of an IPUMS data collection"""
        self.api_version = (
            self.extract_api_version(kwargs)
            if len(kwargs.keys()) > 0
            else self.api_version
        )
        """IPUMS API version number"""
        # check kwargs for conflicts with defaults
        self._kwarg_warning(kwargs)
        # make the kwargs camelCase
        self.kwargs = self._snake_to_camel(kwargs)

    def build(self) -> Dict[str, Any]:
        """
        Convert the object into a dictionary to be passed to the IPUMS API
        as a JSON string
        """
        return {
            "description": self.description,
            "dataFormat": self.data_format,
            "dataStructure": self.data_structure,
            "samples": {sample.id: {} for sample in self.samples},
            "variables": {
                variable.name.upper(): variable.build() for variable in self.variables
            },
            "collection": self.collection,
            "version": self.api_version,
            **self.kwargs,
        }


class CpsExtract(BaseExtract, collection="cps"):
    def __init__(
        self,
        samples: Union[List[str], List[Sample]],
        variables: Union[List[str], List[Variable]],
        description: str = "My IPUMS CPS extract",
        data_format: str = "fixed_width",
        data_structure: Dict = {"rectangular": {"on": "P"}},
        **kwargs,
    ):
        """
        Defining an IPUMS CPS extract.

        Args:
            samples: list of IPUMS CPS sample IDs
            variables: list of IPUMS CPS variable names
            description: short description of your extract
            data_format: fixed_width and csv supported
            data_structure: nested dict with "rectangular" or "hierarchical" as first-level key.
                            "rectangular" extracts require further specification of "on" : <record type>.
                            Default {"rectangular": "on": "P"} requests an extract rectangularized on the "P" record.
        """

        super().__init__()
        self.samples = self._validate_list_args(samples, Sample)
        self.variables = self._validate_list_args(variables, Variable)
        self.description = description
        self.data_format = data_format
        self.data_structure = data_structure
        self.collection = self.collection
        """Name of an IPUMS data collection"""
        self.api_version = (
            self.extract_api_version(kwargs)
            if len(kwargs.keys()) > 0
            else self.api_version
        )
        """IPUMS API version number"""

        # check kwargs for conflicts with defaults
        self._kwarg_warning(kwargs)
        # make the kwargs camelCase
        self.kwargs = self._snake_to_camel(kwargs)

    def build(self) -> Dict[str, Any]:
        """
        Convert the object into a dictionary to be passed to the IPUMS API
        as a JSON string
        """
        return {
            "description": self.description,
            "dataFormat": self.data_format,
            "dataStructure": self.data_structure,
            "samples": {sample.id: {} for sample in self.samples},
            "variables": {
                variable.name.upper(): variable.build() for variable in self.variables
            },
            "collection": self.collection,
            "version": self.api_version,
            **self.kwargs,
        }


class IpumsiExtract(BaseExtract, collection="ipumsi"):
    def __init__(
        self,
        samples: Union[List[str], List[Sample]],
        variables: Union[List[str], List[Variable]],
        description: str = "My IPUMS International extract",
        data_format: str = "fixed_width",
        data_structure: Dict = {"rectangular": {"on": "P"}},
        **kwargs,
    ):
        """
        Defining an IPUMS International extract.

        Args:
            samples: list of IPUMS International sample IDs
            variables: list of IPUMS International variable names
            description: short description of your extract
            data_format: fixed_width and csv supported
            data_structure: nested dict with "rectangular" or "hierarchical" as first-level key.
                            "rectangular" extracts require further specification of "on" : <record type>.
                            Default {"rectangular": "on": "P"} requests an extract rectangularized on the "P" record.
        """

        super().__init__()
        self.samples = self._validate_list_args(samples, Sample)
        self.variables = self._validate_list_args(variables, Variable)
        self.description = description
        self.data_format = data_format
        self.data_structure = data_structure
        self.collection = self.collection
        """Name of an IPUMS data collection"""
        self.api_version = (
            self.extract_api_version(kwargs)
            if len(kwargs.keys()) > 0
            else self.api_version
        )
        """IPUMS API version number"""

        # check kwargs for conflicts with defaults
        self._kwarg_warning(kwargs)
        # make the kwargs camelCase
        self.kwargs = self._snake_to_camel(kwargs)

    def build(self) -> Dict[str, Any]:
        """
        Convert the object into a dictionary to be passed to the IPUMS API
        as a JSON string
        """
        return {
            "description": self.description,
            "dataFormat": self.data_format,
            "dataStructure": self.data_structure,
            "samples": {sample.id: {} for sample in self.samples},
            "variables": {
                variable.name.upper(): variable.build() for variable in self.variables
            },
            "collection": self.collection,
            "version": self.api_version,
            **self.kwargs,
        }


class NhgisExtract(BaseExtract, collection="nhgis"):

    def __init__(
        self,
        data_format: str = "csv_no_header",
        description: str = "My IPUMS NHGIS extract",
        datasets: Optional[Union[List[str], List[Dataset]]] = [],
        timeSeriesTables: Optional[Union[List[str], List[TimeSeriesTable]]] = [],
        shapefiles: list = [],
        timeSeriesTableLayout: str = "time_by_column_layout",
        geographicExtents: list = [],
        breakdownAndDataTypeLayout: str = "single_file",
        **kwargs
    ):
        """
        Defining an IPUMS NHGIS extract.

        Args:
            datasets: list of IPUMS NHGIS Dataset objects
            time_series_tables: list of IPUMS NHGIS TimeSeriesTable objects
            shapefiles: list of IPUMS NHGIS ShapeFile objects
            description: short description of your extract
            data_format: fixed_width, csv_header, and csv_no_header supported. Contrary to name,
                csv_no_header still provides a minimal header row when any datasets or time series tables are selected.
            breakdown_and_data_type_layout: separate_files (split up each data type or breakdown combo into its own file),
                and single_file (keep all data types and breakdowns in a single file)
            time_series_table_layout: time_by_column_layout, time_by_row_layout, or time_by_file_layout
            geographics_extents: list of geographic extents to include in extract
        """

        super().__init__()


        self.datasets = self._validate_list_args(datasets, Dataset)
        self.time_series_tables = self._validate_list_args(timeSeriesTables, TimeSeriesTable)
        self.shapefiles = shapefiles

        if len(self.datasets) == 0 and len(self.time_series_tables) == 0 and len(self.shapefiles) == 0:
            raise ValueError("At least one dataset, time series table, or shapefile must be specified.")
        
        self.description = description
        self.data_format = data_format
        self.breakdown_and_data_type_layout = breakdownAndDataTypeLayout
        self.time_series_table_layout = timeSeriesTableLayout
        self.geographic_extents = geographicExtents
        self.collection = self.collection

        """Name of an IPUMS data collection"""
        self.api_version = (
            self.extract_api_version(kwargs)
            if len(kwargs.keys()) > 0
            else self.api_version
        )
        """IPUMS API version number"""

        # check kwargs for conflicts with defaults
        self._kwarg_warning(kwargs)
        # make the kwargs camelCase
        self.kwargs = self._snake_to_camel(kwargs)

    

    def build(self) -> Dict[str, Any]:
        """
        Convert the object into a dictionary to be passed to the IPUMS API
        as a JSON string
        """

        built = {}

        # since we only need 1 of datasets, timeSeriesTables, or shapefiles to be non-empty,
        # we only include dict keys for non-empty lists

        if present_in_extract_info(self.datasets):
            # "datasets": {"dataset name": {}, "dataset name": {} ... }
            datasets = {
                dataset.name: dataset.build() for dataset in self.datasets
            }
            built["datasets"] = datasets
        
        if present_in_extract_info(self.time_series_tables):
            # "timeSeriesTables": {"time series table name": {}, ... }
            time_series_tables = {
                table.name: table.build() for table in self.time_series_tables
            }
            built["timeSeriesTables"] = time_series_tables
            built["timeSeriesTableLayout"] = self.time_series_table_layout
            
        if present_in_extract_info(self.shapefiles):
            built["shapefiles"] = self.shapefiles

        if present_in_extract_info(self.geographic_extents):
            built["geographicExtents"] = self.geographic_extents

        if present_in_extract_info(self.breakdown_and_data_type_layout):
            built["breakdownAndDataTypeLayout"] = self.breakdown_and_data_type_layout

        # include kwargs if they are not empty
        if present_in_extract_info(self.kwargs):
            built.update(**self.kwargs)

        built["description"] = self.description
        built["dataFormat"] = self.data_format
        built["collection"] = self.collection
        built["version"] = self.api_version

        return built
    
    def __str__(self):
        """
        For testing, print the dataset as a JSON string
        """
        return json.dumps(self.build(), indent=4)


def extract_from_dict(dct: Dict[str, Any]) -> Union[BaseExtract, List[BaseExtract]]:
    """
    Convert an extract that is currently specified as a dictionary (usually from a file)
    into a BaseExtract object. If multiple extracts are specified, return a
    List[BaseExtract] objects.

    Args:
        dct: The dictionary specifying the extract(s)

    Returns:
        The extract(s) specified by dct
    """
    if "extracts" in dct:
        # We are returning several extracts
        return [extract_from_dict(extract) for extract in dct["extracts"]]
    if dct["collection"] in BaseExtract._collection_to_extract:
        # some fanciness to make sure sample and variable features
        # are preserved
        # make samples Sample objects
        if isinstance(dct["samples"], dict):
            dct["samples"] = _unpack_samples_dict(dct["samples"])
        else:
            dct["samples"] = [Sample(id=samp) for samp in dct["samples"]]
        # make varibales Variable objects
        if isinstance(dct["variables"], dict):
            dct["variables"] = _unpack_variables_dict(dct["variables"])
        else:
            dct["variables"] = [Variable(name=var) for var in dct["variables"]]
        if isinstance(dct["timeSeriesTables"], dict):
            dct["timeSeriesTables"] = _unpack_time_series_tables_dict(dct["timeSeriesTables"])
        else:
            dct["timeSeriesTables"] = [TimeSeriesTable(name=table) for table in dct["timeSeriesTables"]]
        if isinstance(dct["shapefiles"], dict):
            dct["datasets"] = _unpack_datasets_dict(dct["datasets"])
        else:
            dct["datasets"] = [Dataset(name=dataset) for dataset in dct["datasets"]]

        return BaseExtract._collection_to_extract[dct["collection"]](**dct)

    return OtherExtract(dct["collection"], dct)


def extract_to_dict(extract: Union[BaseExtract, List[BaseExtract]]) -> Dict[str, Any]:
    """
    Convert an extract object to a dictionary (usually to write to a file).
    If multiple extracts are specified, return a dict object.

    Args:
        extract: A submitted IPUMS extract object or list of submitted IPUMS extract objects

    Returns:
        The extract(s) specified as a dictionary
    """
    dct = {}
    if isinstance(extract, list):
        dct["extracts"] = [extract_to_dict(ext) for ext in extract]
        return dct
    try:
        ext = extract.extract_info
        # just retain the definition part
        return ext["extractDefinition"]

    except ValueError:
        raise IpumsExtractNotSubmitted(
            "Extract has not been submitted and so has no json response"
        )


def save_extract_as_json(extract: Union[BaseExtract, List[BaseExtract]], filename: str):
    """
    Convenience method to save an IPUMS extract definition to a json file.

    Args:
        extract: IPUMS extract object or list of IPUMS extract objects
        filename: Path to json file to which to save the IPUMS extract object(s)
    """
    with open(filename, "w") as fileh:
        json.dump(extract_to_dict(extract), fileh, indent=4)


def define_extract_from_json(filename: str) -> Union[BaseExtract, List[BaseExtract]]:
    """
    Convenience method to convert an IPUMS extract definition or definitions stored
    in a json file into a BaseExtract object. If multiple extracts are specified,
    return a List[BaseExtract] objects.

    Args:
        filename: Json file containing IPUMS extract definitions

    Returns:
        The extract(s) specified in the json file
    """
    with open(filename, "r") as fileh:
        extract = extract_from_dict(json.load(fileh))

    return extract

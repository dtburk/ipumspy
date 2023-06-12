# date: 05 june 2023
# author: bendlev (github.com/bendlev)
# purpose: test the nhgis extract building

from ipumspy import IpumsApiClient, NhgisExtract, Dataset, TimeSeriesTable
from pprint import pprint
import requests
import json

key="59cba10d8a5da536fc06b59d00bc4f897b8f40a2b70ccad3905e84b2"

# initialize client API
ipums = IpumsApiClient(key)

# test dataset formatting
dataset = Dataset(name="1988_1997_CBPa",
                #   years=["1988", "1989", "1990", "1991", "1992", "1993", "1994"],
                  years=["1988", "1990", "1991", "1992", "1993", "1994"],
                  breakdown_values=["bs30.si0762", "bs30.si2026"],
                  data_tables=["NT001"],
                  geog_levels=["county"])

dataset2 = Dataset(name="2000_SF1b",
                   data_tables=["NP001A"],
                     geog_levels=["blck_grp"])

print(dataset)
print("\n")

print(dataset2)
print("\n")

# test time series table formatting
time_series1 = TimeSeriesTable(name="A00",
                               geog_levels=["state"],
                               years=["1990"])

print(time_series1)
print("\n")

# test nhgis extract building
# extract = NhgisExtract(
#     data_format="csv_header",
#     description="Wait, download_extract doesn't work quite yet",
#     datasets=[dataset, dataset2],
#     time_series_tables=[time_series1],
#     time_series_table_layout="time_by_file_layout",
#     shapefiles=["us_state_1790_tl2000"],
#     geographics_extents=["010"],
#     breakdown_and_data_type_layout="single_file",
#     version=2)

x = NhgisExtract(
    datasets = [Dataset("1990_STF1", ["NP1"], ["county"])],
    time_series_tables = [TimeSeriesTable("A00", ["state"])],
    shapefiles = [],
    breakdown_and_data_type_layout="single_file"
)

# built = str(extract.build())

# url = "https://api.ipums.org/extracts?collection=nhgis&version=2"
headers = {"Authorization": key}

# result = requests.post(url, headers=headers, json=json.loads(built))

# extract_number = result.json()["number"]

# pprint(result.json())

# r = requests.get(
#     "https://api.ipums.org/extracts/6?collection=nhgis&version=2",
#     headers=headers
# )

# extract = r.json()

# extract_links = extract["downloadLinks"]

# for link in extract_links:
#     print(link, "\t", extract_links[link])

# r = requests.get(extract_links["tableData"]["url"], allow_redirects=True,
#                  headers=headers)

# open("/home/bendy/github/ipumspy/tests/nhgis0006_csv.zip", "wb").write(r.content)

# print("\n\n")
# print(built)
# print("\n\n")
# for b in built:
#     print(b, built[b])

# # big kahuna, does it work?
numb = ipums.submit_extract(extract, collection="nhgis")

# # # get extract status
# status = ipums.extract_status(extract)
# print(status)

# # # wait for extract to finish
ipums.wait_for_extract(numb)

# download extract
ipums.download_extract(extract, download_dir="/home/bendy/github/ipumspy/tests/test_download/")

# print(ipums.get_extract_info("6", collection="nhgis"))

# download extract from link

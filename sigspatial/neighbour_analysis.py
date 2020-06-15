"""
 Author: Luana Fragoso
 Script to analyze participant's movement based on their neighourhood with INTERACT dataset in Victoria city
"""

import pandas as pd
import fiona
import similarity as stable
import analysis as stplt

from shapely.geometry import shape, Point
from geopy.geocoders import Nominati

import importlib
importlib.reload(stable)
importlib.reload(stplt)


file = open("info", "r").readlines()
HEALTH_SURVEY_PATH = file[0].rstrip()
VICTORIA_DATA_PATH = file[1].rstrip()
RESULTS_PATH = file[2].rstrip() + "/neighbour/"
# shape file from http://opendata.victoria.ca/datasets/assessment-values-by-neighbourhood-and-property-type/
SHAPEFILE_PATH = file[3].rstrip()
VERITA_SURVEY_PATH = file[4].rstrip()


def find_neighbourhood(row, multipol):
  for multi in multipol:
    if Point(row["pos_y"], row["pos_x"]).within(shape(multi["geometry"])) or \
       Point(row["pos_y"], row["pos_x"]).touches(shape(multi["geometry"])):
      return multi["properties"]["Neighbourh"]
    else:
      return "Out Victoria"


def get_data():
  vic_min = pd.read_csv(VICTORIA_DATA_PATH)
  vic_min = vic_min[~vic_min["lat"].isna()]
  vic_min = vic_min[["utcdate", "interact_id", "easting", "northing"]]
  vic_min.columns = ["timestamp", "id", "pos_x", "pos_y"]

  vic_verita = pd.read_csv(VERITA_SURVEY_PATH)
  vic_verita = vic_verita[vic_verita["interact_id"].isin(pd.unique(vic_min["id"]))]

  # for i in airports:
  #   loc = geolocator.geocode(i + " airport")

  # Code for checking if a point is in a polygon inspired by:
  # https://stackoverflow.com/questions/43892459/check-if-geo-point-is-inside-or-outside-of-polygon
  # https://gis.stackexchange.com/questions/208546/check-if-a-point-falls-within-a-multipolygon-with-python
  multipol = fiona.open(SHAPEFILE_PATH)

  vic_min["neighbourh"] = vic_min.apply(find_neighbourhood, axis=1, args=(multipol,))

  return vic_min


# How similar are movements btw neighbourhoods?
M = get_data()
M = M[M["neighbourh"] != "Out Victoria"]
M.columns = ["timestamp", "uid", "pos_x", "pos_y", "id"]
c_type = stable.Construction.dwell_trip
I = pd.DataFrame({"group": pd.unique(M["id"])})

sims = stable.quantify(M, I.index, c_type, 150, max_n=10)
ngram = "min_n"

# -------- plots
stplt.plot_boxplot(RESULTS_PATH, sims["btw"][ngram], I["group"], c_type, c_type.value, add_id=False)
# stat_test(sims["btw"]["min_n"], I)
stplt.plot_bar(RESULTS_PATH, [len(sim) for sim in sims["btw"]["features"]], I["group"], c_type)

# -------- graph analysis
threshold = 0.1

G = stplt.to_graph(sims["btw"][ngram], I["group"].index, threshold)
stplt.plot_graph(RESULTS_PATH, G, I["group"], threshold, c_type, c_type.value)

partitions = stplt.louvain(sims["btw"][ngram], I["group"], threshold, two_partitions=False)
stplt.plot_graph(RESULTS_PATH, G, partitions["partition"], threshold, c_type, 10)


# How similar are participant movements in the same neighbourhood?
# Victoria West
# Fairfield
# South Jubilee
VW = M[M["id"] == "Fairfield"]
VW.columns = ["timestamp", "id", "pos_x", "pos_y", "nid"]
I = pd.DataFrame({"id": pd.unique(VW["id"]), "group": [0] * len(pd.unique(VW["id"]))})

c_type = stable.Construction.dwell_trip
sims = stable.quantify(VW, I["id"].values, c_type, 150, max_n=10)
ngram = "min_n"

# -------- plots
stplt.plot_boxplot(RESULTS_PATH, sims["btw"][ngram], I["group"], c_type, c_type.value, add_id=False)
# stat_test(sims["btw"]["min_n"], I)
stplt.plot_bar(RESULTS_PATH, [len(sim) for sim in sims["btw"]["features"]], I["group"], c_type)

# -------- graph analysis
threshold = 0.5

G = stplt.to_graph(sims["btw"][ngram], I["group"].index, threshold)
stplt.plot_graph(RESULTS_PATH, G, I["group"], threshold, c_type, c_type.value)

partitions = stplt.louvain(sims["btw"][ngram], I["group"], threshold, two_partitions=False)
stplt.plot_graph(RESULTS_PATH, G, partitions["partition"], threshold, c_type, 10)





# How similar are participant's movement among different neighbourhoods?
# How similar are participant's movement weekday x weekend?

# Hypothesis
# Smaller cities would have more similar movements because there aren't many options of routes compared to bigger cities

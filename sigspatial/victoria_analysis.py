########################
# Author: Luana Fragoso
# Script to analyze participant's movement with INTERACT dataset in Victoria city
########################

import pandas as pd

import similarity as stable
import importlib
importlib.reload(stable)


HEALTH_SURVEY_PATH = ""
VICTORIA_DATA_PATH = ""


def classify_cyclists():
    vic_health = pd.read_csv(HEALTH_SURVEY_PATH)
    vic_health = vic_health[vic_health["interact_id"].isin(pd.unique(vic_min["id"]))]

    cycl = vic_health[vic_health["bike_freq_a"] > 50]
    cycl = cycl[cycl["bike_freq_c"] > 50]
    cycl = cycl[cycl["bike_freq_d"] > 50]
    cycl = cycl[["interact_id"]]
    cycl["cyclist"] = 1

    non_cycl = vic_health[~vic_health.isin(cycl)]
    non_cycl = non_cycl[~non_cycl["interact_id"].isna()]
    non_cycl = non_cycl[["interact_id"]]
    non_cycl["cyclist"] = 0

    df = cycl.append(non_cycl)
    df.columns = ["id", "cyclist"]

    return df


vic_min = pd.read_csv(VICTORIA_DATA_PATH)
vic_min = vic_min[~vic_min["lat"].isna()]
vic_min = vic_min[["utcdate", "interact_id", "easting", "northing"]]
vic_min.columns = ["timestamp", "id", "pos_x", "pos_y"]

info = classify_cyclists()

sims = stable.quantify(vic_min, info.iloc[0:1], stable.Construction.dwell_trip, 100)



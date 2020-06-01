"""
 Author: Luana Fragoso
 Script to analyze participant's movement with INTERACT dataset in Victoria city
"""

import pandas as pd
import numpy as np
import scikit_posthocs as sp

import similarity as stable
import analysis as stplt

import importlib
importlib.reload(stable)
importlib.reload(stplt)

from scipy import stats

file = open("info", "r").readlines()
HEALTH_SURVEY_PATH = file[0].rstrip()
VICTORIA_DATA_PATH = file[1].rstrip()
RESULTS_PATH = file[2].rstrip()


def classify_cyclists():
    vic_health = pd.read_csv(HEALTH_SURVEY_PATH)
    vic_health = vic_health[vic_health["interact_id"].isin(pd.unique(vic_min["id"]))]
    vic_health = vic_health[["interact_id", "bike_freq_a", "bike_freq_b", "bike_freq_c", "bike_freq_d"]]

    # Use quantile to create bins with a more balanced number of data
    fall = vic_health["bike_freq_a"].quantile([.33, .66]).values
    winter = vic_health["bike_freq_b"].quantile([.33, .66]).values
    spring = vic_health["bike_freq_c"].quantile([.33, .66]).values
    summer = vic_health["bike_freq_d"].quantile([.33, .66]).values

    # It is hard to find a participant with all the season sastifying the biking frequency quantiles.
    # Instead, we take the frequency mode per participant as the final cyclist classification.
    vic_health["fall_freq"] = ""
    vic_health.loc[vic_health["bike_freq_a"] <= fall[0], "fall_freq"] = 0
    vic_health.loc[((vic_health["bike_freq_a"] > fall[0]) &
                    (vic_health["bike_freq_a"] <= fall[1])), "fall_freq"] = 1
    vic_health.loc[vic_health["fall_freq"] == "", "fall_freq"] = 2

    vic_health["winter_freq"] = ""
    vic_health.loc[vic_health["bike_freq_b"] <= winter[0], "winter_freq"] = 0
    vic_health.loc[((vic_health["bike_freq_b"] > winter[0]) &
                    (vic_health["bike_freq_b"] <= winter[1])), "winter_freq"] = 1
    vic_health.loc[vic_health["winter_freq"] == "", "winter_freq"] = 2

    vic_health["spring_freq"] = ""
    vic_health.loc[vic_health["bike_freq_c"] <= spring[0], "spring_freq"] = 0
    vic_health.loc[((vic_health["bike_freq_c"] > spring[0]) &
                    (vic_health["bike_freq_c"] <= spring[1])), "spring_freq"] = 1
    vic_health.loc[vic_health["spring_freq"] == "", "spring_freq"] = 2

    vic_health["summer_freq"] = ""
    vic_health.loc[vic_health["bike_freq_d"] <= summer[0], "summer_freq"] = 0
    vic_health.loc[((vic_health["bike_freq_d"] > summer[0]) &
                    (vic_health["bike_freq_d"] <= summer[1])), "summer_freq"] = 1
    vic_health.loc[vic_health["summer_freq"] == "", "summer_freq"] = 2

    modes = vic_health[["fall_freq", "winter_freq", "spring_freq", "summer_freq"]].mode(axis="columns").values

    classes = {0: "Low", 1: "Medium", 2: "High"}
    groups = []
    for m in modes:
        if not np.isnan(m).any():
            groups.append(m.max())
        else:
            groups.append(m[0])
    vic_health["group"] = groups

    df = vic_health[["interact_id", "group"]]
    df.columns = ["id", "group"]
    df = df.sort_values(by="group")
    df["group"] = df["group"].map(classes)

    return df.reset_index(drop=True)


def stat_test(sims, labels, btw=True):
    if btw:
        between = []
        for i in labels.index:
            for j in sims[i]:
                between.append({'id': i, 'score': j, 'group': labels.loc[i, 'group']})

        # take the mean to wash out the dependency between individuals and be able to use the kruskal test
        between = pd.DataFrame(between).groupby(['id'], as_index=False).agg({'group': 'first', 'score': 'mean'}) \
            .sort_values(['id'])

        low = between.loc[between["group"] == "Low", 'score']
        medium = between.loc[between["group"] == "Medium", 'score']
        high = between.loc[between["group"] == "High", 'score']

        stat, p = stats.kruskal(low, medium, high)

        if p <= 0.05:
            posthoc = sp.posthoc_dunn([low, medium, high])
            return p, posthoc
    else:
        withins = []
        for i in range(0, len(sims)):
            withins.append((labels.iloc[i]["group"], sims[i]))

        withins = pd.DataFrame(withins)
        withins.columns = ["group", "score"]

        low = withins.loc[withins["group"] == "Low", 'score']
        medium = withins.loc[withins["group"] == "Medium", 'score']
        high = withins.loc[withins["group"] == "High", 'score']

        stat, p = stats.kruskal(low, medium, high)

        if p <= 0.05:
            posthoc = sp.posthoc_dunn([low, medium, high])
            return p, posthoc


vic_min = pd.read_csv(VICTORIA_DATA_PATH)
vic_min = vic_min[~vic_min["lat"].isna()]
vic_min = vic_min[["utcdate", "interact_id", "easting", "northing"]]
vic_min.columns = ["timestamp", "id", "pos_x", "pos_y"]

info = classify_cyclists()

c_type = stable.Construction.dwell_trip
# 2.5km for cell size since biking usually require longer travels in which walking would take too long
# 2.5km has an average of 30min walking, where biking would take on avg 7.5min
sims = stable.quantify(vic_min, info["id"].values, c_type, 2500)

# -------- plots
stplt.plot_boxplot(RESULTS_PATH, sims["btw"]["min_n"], info["group"], c_type, c_type.value, add_id=False)
stat_test(sims["btw"]["min_n"], info)

stplt.plot_bar(RESULTS_PATH, sims["wth"]["min_n"], info["group"], c_type, min_ngram=c_type.value, add_id=False)
stat_test(sims["wth"]["min_n"], info, btw=False)

# Within and between similarity analysis did not show any difference between the similarity distributions of
# biking frequency groups

# -------- graph analysis
threshold = 0.6
G = stplt.to_graph(sims["btw"]["min_n"], info["group"].index, threshold)
stplt.plot_graph(RESULTS_PATH, G, info["group"], threshold, c_type, c_type.value, node_label=True)

partitions = stplt.louvain(sims["btw"]["min_n"], info["group"], threshold, two_partitions=False)
stplt.plot_graph(RESULTS_PATH, G, partitions["partition"], threshold, c_type, c_type.value)

# Threshold 0 - Why participant 54 is separated from the rest?
# Threshold 0.1 - Why participant 13 and 118 are separated from the rest?
# Threshold 0.4 - Are participants 127 and 135 from same family/neighbour?
# Threshold 0.5 - Are participants 124, 64, 69, 40, and 17 from the same family/neighbour?
# Threshold 0.6 - Why participants 124 and 40 are the most similar pair in the data?





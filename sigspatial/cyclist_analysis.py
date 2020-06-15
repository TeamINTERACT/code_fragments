"""
 Author: Luana Fragoso
 Script to analyze participant's movement based on their cycling frequency with INTERACT dataset in Victoria city
"""

import folium
import pandas as pd
import scikit_posthocs as sp

import similarity as stable
import analysis as stplt

from scipy import stats
from folium.plugins import HeatMap

import importlib
importlib.reload(stable)
importlib.reload(stplt)


file = open("info", "r").readlines()
HEALTH_SURVEY_PATH = file[0].rstrip()
VICTORIA_DATA_PATH = file[1].rstrip()
RESULTS_PATH = file[2].rstrip() + "/cyclist/"
TOP_PATH = file[5].rstrip()
VERITA_SURVEY_PATH = file[4].rstrip()


def get_data():
    # vic_min = pd.read_csv(VICTORIA_DATA_PATH)
    # vic_min = vic_min[~vic_min["lat"].isna()]
    # vic_min = vic_min[["utcdate", "interact_id", "easting", "northing", "lat", "lon"]]
    # vic_min.columns = ["timestamp", "id", "pos_x", "pos_y", "lat", "lon"]
    # vic_min["month"] = pd.DatetimeIndex(vic_min['timestamp']).month
    # vic_min = vic_min[vic_min["month"].isin([5, 6, 7, 8, 9])]

    vic_min = pd.read_csv(TOP_PATH)
    vic_min = vic_min[~vic_min["lat"].isna()]
    vic_min = vic_min[["utcdate", "interact_id", "easting", "northing", "activity_levels"]]
    vic_min.columns = ["timestamp", "id", "pos_x", "pos_y", "action"]
    vic_min["month"] = pd.DatetimeIndex(vic_min['timestamp']).month
    vic_min = vic_min[vic_min["month"].isin([6, 7, 8, 9])]

    vic_health = pd.read_csv(VERITA_SURVEY_PATH)
    vic_health = vic_health[["interact_id", "hours_out_neighb"]]
    vic_health = vic_health.dropna()
    vic_health.columns = ["id", "group"]
    vic_health.loc[vic_health["group"] < 4, "group"] = 0
    vic_health.loc[((vic_health["group"] >= 4) & (vic_health["group"] <= 8)), "group"] = 1
    vic_health.loc[vic_health["group"] > 8, "group"] = 2
    vic_health = vic_health.sort_values(by=["group"])

    vic_health = vic_health[vic_health["id"].isin(pd.unique(vic_min["id"]))]
    vic_min = vic_min[vic_min["id"].isin(pd.unique(vic_health["id"]))]

    return vic_min.reset_index(drop=True), vic_health.reset_index(drop=True)


def save_heatmap():
    m = folium.Map(location=[48.3495341, -123.7015091], zoom_start=6)
    HeatMap(M[["lat", "lon"]].values).add_to(m)
    m.save(RESULTS_PATH + "heat_map.html")


def stat_test(sims, labels, btw=True):
    if btw:
        between = []
        for i in labels.index:
            for j in sims[i]:
                between.append({'id': i, 'score': j, 'group': labels.loc[i, 'group']})

        # take the mean to wash out the dependency between individuals and be able to use the kruskal test
        between = pd.DataFrame(between).groupby(['id'], as_index=False).agg({'group': 'first', 'score': 'mean'}) \
            .sort_values(['id'])

        g1 = between.loc[between["group"] == 0, 'score']
        g2 = between.loc[between["group"] == 1, 'score']
        g3 = between.loc[between["group"] == 2, 'score']

        stat, p = stats.kruskal(g1, g2, g3)

        # stat, p = stats.mannwhitneyu(g1, g2)

        return p

        # low = between.loc[between["group"] == "Low", 'score']
        # medium = between.loc[between["group"] == "Medium", 'score']
        # high = between.loc[between["group"] == "High", 'score']
        #
        # stat, p = stats.kruskal(low, medium, high)
        #
        # if p <= 0.05:
        #     posthoc = sp.posthoc_dunn([low, medium, high])
        #     return p, posthoc
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


def louvain_stat(labels, dependent):
    merged = labels.merge(dependent["group"], how='left', left_index=True, right_index=True)
    merged.columns = ['partition', "group"]

    n_part = max(merged['partition'])

    if n_part == 1:
        p0 = merged.loc[merged['partition'] == 0, "group"].values
        p1 = merged.loc[merged['partition'] == 1, "group"].values

        F, p = stats.mannwhitneyu(p0, p1)

    elif n_part == 2:
        p0 = merged.loc[merged['partition'] == 0, "group"].values
        p1 = merged.loc[merged['partition'] == 1, "group"].values
        p2 = merged.loc[merged['partition'] == 2, "group"].values

        F, p = stats.kruskal(p0, p1, p2)

    elif n_part == 3:
        p0 = merged.loc[merged['partition'] == 0, "group"].values
        p1 = merged.loc[merged['partition'] == 1, "group"].values
        p2 = merged.loc[merged['partition'] == 2, "group"].values
        p3 = merged.loc[merged['partition'] == 3, "group"].values

        F, p = stats.kruskal(p0, p1, p2, p3)

    elif n_part == 4:
        p0 = merged.loc[merged['partition'] == 0, "group"].values
        p1 = merged.loc[merged['partition'] == 1, "group"].values
        p2 = merged.loc[merged['partition'] == 2, "group"].values
        p3 = merged.loc[merged['partition'] == 3, "group"].values
        p4 = merged.loc[merged['partition'] == 4, "group"].values

        F, p = stats.kruskal(p0, p1, p2, p3, p4)

    elif n_part == 5:
        p0 = merged.loc[merged['partition'] == 0, "group"].values
        p1 = merged.loc[merged['partition'] == 1, "group"].values
        p2 = merged.loc[merged['partition'] == 2, "group"].values
        p3 = merged.loc[merged['partition'] == 3, "group"].values
        p4 = merged.loc[merged['partition'] == 4, "group"].values
        p5 = merged.loc[merged['partition'] == 5, "group"].values

        F, p = stats.kruskal(p0, p1, p2, p3, p4, p5)

    elif n_part == 6:
        p0 = merged.loc[merged['partition'] == 0, "group"].values
        p1 = merged.loc[merged['partition'] == 1, "group"].values
        p2 = merged.loc[merged['partition'] == 2, "group"].values
        p3 = merged.loc[merged['partition'] == 3, "group"].values
        p4 = merged.loc[merged['partition'] == 4, "group"].values
        p5 = merged.loc[merged['partition'] == 5, "group"].values
        p6 = merged.loc[merged['partition'] == 6, "group"].values

        F, p = stats.kruskal(p0, p1, p2, p3, p4, p5, p6)

    return F, p


M, I = get_data()
# save_heatmap()

c_type = stable.Construction.dwell_trip_action
# 150m because people bike at the slowest on an avg of 10km/h. Because our data is in minutes, this same person would
# travel around 166m in a minute. 150m is a reasonable simplified round.
sims = stable.quantify(M, I["id"].values, c_type, 150, max_n=10)
ngram = "min_n"

# -------- plots
stplt.plot_boxplot(RESULTS_PATH, sims["btw"][ngram], I["group"], c_type, c_type.value, add_id=False)
stat_test(sims["btw"][ngram], I)

stplt.plot_bar(RESULTS_PATH, sims["wth"][ngram], I["group"], c_type, min_ngram=c_type.value, add_id=False)
stat_test(sims["wth"]["min_n"], I, btw=False)


# -------- graph analysis
threshold = 0.3

G = stplt.to_graph(sims["btw"][ngram], I["group"].index, threshold)
stplt.plot_graph(RESULTS_PATH, G, I["group"], threshold, c_type, c_type.value)

partitions = stplt.louvain(sims["btw"][ngram], I["group"], threshold, two_partitions=False)
stplt.plot_graph(RESULTS_PATH, G, partitions["partition"], threshold, c_type, 10)
F, p = louvain_stat(partitions, I)

# n-range 4 to 8

# age - 500m
# 0.1 -> p < 0.009





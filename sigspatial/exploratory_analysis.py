import pandas as pd
import matplotlib.pyplot as plt
import prince
import similarity as stable
import analysis as stplt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


import importlib
importlib.reload(stable)
importlib.reload(stplt)

file = open("info", "r").readlines()
ELIGIBILITY_SURVEY_PATH = file[6].rstrip()
HEALTH_SURVEY_PATH = file[0].rstrip()
VERITA_SURVEY_PATH = file[4].rstrip()
SENSEDOC_PATH = file[1].rstrip()
TOP_PATH = file[5].rstrip()
RESULTS_PATH = file[2].rstrip() + "/exploratory/"


def plot_pca(partitions):
  comp = vic[vic["id"].isin(partitions["id"])].sort_values(by=["id"])
  comp = comp.iloc[:, 1:]
  comp = comp.select_dtypes(exclude="object")
  # Uncomment to check if any column has missing values
  # [(col, len(comp[comp[col].isna()])) for col in comp.columns[comp.isna().any()].tolist()]
  comp = comp.drop(columns=["work_pa"])  # this column has many missing values
  comp = comp.dropna()  # other columns still have a few missing values

  # Normalize data with no missing values
  mapper = DataFrameMapper([(comp.columns, StandardScaler())])
  comp = pd.DataFrame(mapper.fit_transform(comp.copy()), index=comp.index, columns=comp.columns)
  partitions = partitions.loc[comp.index]

  pca = PCA(n_components=2)
  principalComponents = pca.fit_transform(comp)
  pca_df = pd.DataFrame(data=principalComponents, columns=["pca1", "pca2"])
  pca_df = pd.concat([pca_df, partitions[['partition']].reset_index()], axis=1)
  colors = stplt.define_colors(partitions["partition"])
  pca_df = pd.concat([pca_df, pd.DataFrame({"color": colors.values})], axis=1)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_xlabel('Principal Component 1 (' + str(round(pca.explained_variance_ratio_[0] * 100, 2)) + '%)', fontsize=15)
  ax.set_ylabel('Principal Component 2 (' + str(round(pca.explained_variance_ratio_[1] * 100, 2)) + '%)', fontsize=15)
  ax.set_title('2 component PCA', fontsize=20)

  for color in pd.unique(pca_df["color"]):
    indicesToKeep = pca_df['color'] == color
    ax.scatter(pca_df.loc[indicesToKeep, 'pca1'],
               pca_df.loc[indicesToKeep, 'pca2'],
               c=[color] * len(pca_df.loc[indicesToKeep]), s=50)
  ax.legend(["0", "1"])
  fig.savefig(RESULTS_PATH + 'pca.png')


def plot_mca(partitions):
  comp = vic[vic["id"].isin(partitions["id"])].sort_values(by=["id"])
  comp = comp.iloc[:, 1:]
  comp = comp.select_dtypes(include="object")
  # Uncomment to check if any column has missing values
  # [(col, len(comp[comp[col].isna()])) for col in comp.columns[comp.isna().any()].tolist()]
  comp = comp.drop(columns=["preferred_mode_f_txt", "car_share_txt", "house_tenure_txt", "dwelling_type_txt",
                            "living_arrange_txt", "home_address_line2"])  # this column has many missing values
  comp = comp.dropna()  # other columns still have a few missing values
  partitions = partitions.loc[comp.index]

  mca = prince.MCA()
  mca = mca.fit(comp)  # same as calling ca.fs_r(1)
  mca_df = mca.transform(comp).reset_index(drop=True)  # same as calling ca.fs_r_sup(df_new) for *another* test set.
  mca_df = pd.concat([mca_df, partitions[['partition']].reset_index()], axis=1)
  colors = stplt.define_colors(partitions["partition"])
  mca_df = pd.concat([mca_df, pd.DataFrame({"color": colors.values})], axis=1)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_xlabel('Component 1 (' + str(round(mca.explained_inertia_[0] * 100, 2)) + '%)', fontsize=15)
  ax.set_ylabel('Component 2 (' + str(round(mca.explained_inertia_[1] * 100, 2)) + '%)', fontsize=15)
  ax.set_title('2 component MCA', fontsize=20)

  for color in pd.unique(mca_df["color"]):
    indicesToKeep = mca_df['color'] == color
    ax.scatter(mca_df.loc[indicesToKeep, 0],
               mca_df.loc[indicesToKeep, 1],
               c=[color] * len(mca_df.loc[indicesToKeep]), s=50)
  ax.legend(["0", "1"])
  fig.savefig(RESULTS_PATH + 'mca.png')


def plot_decision_tree(partitions):
  comp = vic[vic["id"].isin(partitions["id"])].sort_values(by=["id"])
  comp = comp[["age", "gender_vic_x", "children_household", "bike_more", "cars_household", "bike_comf_a", "bike_comf_f",
               "house_tenure", "travel_motor", "travel_bike", "travel_walk", "sf1", "marital_status", "residence",
               "income_needs"]]

  gender = {"[2]": 2, "[1]": 1, "[4]": 3, "[1, 4]": 4}
  comp["gender_vic_x"] = comp["gender_vic_x"].map(gender)

  comp['residence'] = pd.to_datetime(comp.residence).apply(lambda x: x.year)
  comp['residence'] = 2020 - comp['residence']

  # comp = comp.select_dtypes(exclude="object")
  # Uncomment to check if any column has missing values
  # [(col, len(comp[comp[col].isna()])) for col in comp.columns[comp.isna().any()].tolist()]
  # comp = comp.drop(columns=["work_pa"])  # this column has many missing values
  # comp = comp.dropna()  # other columns still have a few missing values
  partitions = partitions.loc[comp.index]

  clf = DecisionTreeClassifier(random_state=0)
  clf.fit(comp, partitions["partition"])

  fig, ax = plt.subplots(figsize=(40, 10))
  tree.plot_tree(clf,
                 feature_names=comp.columns.values,
                 class_names=pd.unique(partitions["partition"]).astype(str),
                 filled=True, fontsize=8)
  # plt.show()
  fig.savefig(RESULTS_PATH + 'decision-tree.png', dpi=100)


def get_data():
  elig = pd.read_csv(ELIGIBILITY_SURVEY_PATH)
  health = pd.read_csv(HEALTH_SURVEY_PATH)
  verita = pd.read_csv(VERITA_SURVEY_PATH)
  top = pd.read_csv(TOP_PATH)

  top = top[~top["lat"].isna()]
  vic = elig.merge(health, on="interact_id").merge(verita, on="interact_id")
  vic = vic[vic["interact_id"].isin(pd.unique(top["interact_id"]))]
  vic.rename(columns={'interact_id': 'id'}, inplace=True)

  top = top[["utcdate", "interact_id", "easting", "northing", "activity_levels"]]
  top.columns = ["timestamp", "id", "pos_x", "pos_y", "action"]
  top = top[top["id"].isin(pd.unique(vic["id"]))]

  return top, vic


def weekday_weekend():
  weekday = []
  weekend = []
  for p in I["id"].values:
    mov = M[M["id"] == p]
    mov = mov.rename(columns={"id": "uid"})
    mov["id"] = pd.to_datetime(mov["timestamp"]).dt.dayofweek

    weekday.append(stable.quantify(mov, pd.unique(mov.loc[mov["id"] < 5, "id"].sort_values()), c_type, grid,
                                   max_n=max_n))
    weekend.append(stable.quantify(mov, pd.unique(mov.loc[mov["id"] > 4, "id"].sort_values()), c_type, grid,
                                   max_n=max_n))

  return weekday, weekend


# =================== #
# Clustering Analyses #
# =================== #
grid_sizes = [25, 50, 100, 200, 400, 800, 1600, 3200]
max_ns = [4, 6, 8, 10]
c_type = stable.Construction.dwell_trip

M, vic = get_data()
I = pd.DataFrame({"id": vic["id"].copy()})
ngram = "min_n"

for grid in grid_sizes:
  for max_n in max_ns:
    sims = stable.quantify(M, I["id"], c_type, grid, max_n=max_n)

    # -------- graph analysis
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for threshold in thresholds:
      try:
        G = stplt.to_graph(sims["btw"][ngram], I.index, threshold)
        partitions = stplt.louvain(sims["btw"][ngram], I, threshold, two_partitions=False)
        partitions = partitions.merge(I, how='left', left_index=True, right_index=True).sort_values(by=["id"])
        stplt.plot_graph(RESULTS_PATH + str(grid) + "/", G, partitions["partition"], threshold, c_type, max_n)
      except:
        print("Not many clusters")

# -------- degree analysis
grid = 25
max_n = 10
sims = stable.quantify(M, I["id"], c_type, grid, max_n=max_n)

threshold = 0.4
G = stplt.to_graph(sims["btw"][ngram], I.index, threshold)
partitions = stplt.louvain(sims["btw"][ngram], I, threshold, two_partitions=False)
partitions = partitions.merge(I, how='left', left_index=True, right_index=True).sort_values(by=["id"])
stplt.plot_graph(RESULTS_PATH + str(grid) + "/", G, partitions["partition"], threshold, c_type, max_n)

degrees = pd.DataFrame({"node": list(dict(G.degree()).keys()), "degree": list(dict(G.degree()).values())})
partitions = partitions.merge(degrees, how="left", left_index=True, right_on=["node"])
partitions = partitions.set_index(partitions["node"]).drop(columns="node")
# stplt.plot_bar(RESULTS_PATH + str(grid) + "/", partitions["degree"].values, partitions["partition"], c_type)

# Uncomment and place the partition group to find the key person in the given group
# partitions.loc[partitions["partition"] == 3, "degree"].max()
# partitions.loc[(partitions["partition"] == 3) & (partitions["degree"] == 5), "id"]

# ========================== #
# Clustering-Survey Analyses #
# ========================== #
plot_pca(partitions)
plot_mca(partitions)
plot_decision_tree(partitions)
plot_decision_tree(partitions[partitions["id"].isin([101664312, 101253378, 101751848, 101215885])])


weekday, weekend = weekday_weekend()


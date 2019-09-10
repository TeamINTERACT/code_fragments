'''
Author: Luana Fragoso
Script to analyze participant's behavior with INTERACT dataset in Victoria city
'''

import stable
import community
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def plot_hist(features, title, x_label, log=False):
    plt.figure()

    x = ""
    for f in features:
        x += f + " "

    x = x.rstrip().split()

    plt.hist(x, bins=30)
    if log:
        plt.xscale('log', basex=2)
        plt.yscale('log', basey=2)
    plt.xlabel(x_label)
    plt.title(title + " Distributions")

    plt.savefig("results/hist-" + title + ".png", orientation='portrait', format='png', dpi=1000)


def plot_boxplot(sims, min_ngram, max_ngram, analysis):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)

    if min_ngram == max_ngram:
        plt.title("Similarities " + analysis + " with ngram = " + str(min_ngram), fontsize=17)
    else:
        plt.title("Similarities " + analysis + " with min_ngram = " + str(min_ngram) + " and max_ngram = "
                  + str(max_ngram), fontsize=17)
    plt.ylim(0, 1)
    bp = ax.boxplot(sims, patch_artist=True)

    # Customize the axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    plt.rc('ytick', labelsize=18)

    for i in range(0, len(sims)):
        bp['boxes'][i].set(facecolor='#ababab', linewidth=0.1)
        bp['whiskers'][i * 2].set(linewidth=0.5)
        bp['whiskers'][i * 2 + 1].set(linewidth=0.5)
        bp['caps'][i * 2].set(linewidth=0.5)
        bp['caps'][i * 2 + 1].set(linewidth=0.5)
        bp['medians'][i].set(linewidth=1.5, color='#ff800e')
        bp['fliers'][i].set(marker='o', color='#898989', alpha=0.5, markersize=3, markerfacecolor='#898989',
                           markeredgecolor='#898989')

    plt.ylabel("Similarity Score", fontsize=17)
    plt.xlabel("Participant", fontsize=17)
    plt.savefig('results/sim_boxplot_' + analysis + '_ngram' + str(min_ngram) + '.png', dpi=500)
    plt.close()


def plot_bar(y, min_ngram=None, max_ngram=None, analysis='', sim=True):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    plt.xlabel("Participants ID", fontsize=17)

    ax.bar(range(len(y)), y, color="#006ba4", edgecolor='black', linewidth=0.1)

    # Customize the axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    if sim:
        plt.ylabel("Similarity Score", fontsize=17)
        plt.ylim(0, 1)
        if min_ngram == max_ngram:
            plt.title("Similarities " + analysis + " with ngram = " + str(min_ngram), fontsize=17)
            plt.savefig('results/sim_bar_' + analysis + '_ngram' + str(min_ngram) + '.png', dpi=500)
        else:
            plt.title("Similarities " + analysis + " with min_ngram = " + str(min_ngram) + " and max_ngram = "
                      + str(max_ngram), fontsize=17)
            plt.savefig('results/sim_bar_' + analysis + '_minngram' + str(min_ngram) + '_maxngram'
                        + str(max_ngram) + '.png', dpi=500)
    else:
        plt.ylabel("Frequency", fontsize=17)
        plt.title("Word length per player", fontsize=17)
        plt.savefig('results/word_len.png', dpi=500)
    plt.close()


def plot_graph(G, labels, threshold, ngram):
    pos = nx.kamada_kawai_layout(G)

    color_map = []
    for node in G.nodes.keys():
        if labels.loc[labels['name'] == node, 'partition'].values[0] == 0:
            color_map.append('#ff9e4a')
        if labels.loc[labels['name'] == node, 'partition'].values[0] == 1:
            color_map.append('#ed665d')
        if labels.loc[labels['name'] == node, 'partition'].values[0] == 2:
            color_map.append('#ad8bc9')
        if labels.loc[labels['name'] == node, 'partition'].values[0] == 3:
            color_map.append('#a8786e')

    plt.figure()
    plt.title("Similarities greater than " + str(threshold) + " with ngram = " + str(ngram))
    nx.draw(G, pos, with_labels=True, node_color=color_map, edge_color='grey', font_size=11, width=0.5)

    plt.savefig('results/' + str(threshold) + 'sim_ngram' + str(ngram) + '.png', dpi=500)
    plt.close()


def louvain(sims, participants, threshold, ngram):
    similars = stable.filter_sims(sims, participants, threshold)
    similars = sorted(similars, key=lambda x: x[2])

    if len(similars) > 0:
        G = nx.Graph()
        G.add_weighted_edges_from(similars)
        w_degrees = sorted(G.degree(weight='weight'), key=lambda x: x[0])
        degrees = sorted(G.degree, key=lambda x: x[0])
        degrees = [(w_degrees[i][0], w_degrees[i][1]/degrees[i][1]) for i in range(len(w_degrees))]
        degrees = sorted(degrees, key=lambda x: x[1])


        partitions = {}
        for i in range(len(degrees)-1, -1, -1):
            if i <= (len(degrees)//2):
                partitions[degrees[i][0]] = 0
            else:
                partitions[degrees[i][0]] = 1
        best_partition = community.best_partition(G, randomize=False, partition=partitions)
        partitions = []
        for i in best_partition:
            partitions.append((i, best_partition[i]))
        partitions = pd.DataFrame(partitions, columns=['name', 'partition'])

        plot_graph(G, partitions, threshold, ngram)

        return partitions


file = open("info", "r").readlines()
server = file[0].rstrip()
db = file[1].rstrip()
user = file[2].rstrip()
pssw = file[3].rstrip()
engine = create_engine('postgresql://' + user + ":" + pssw + "@" + server + "/" + db)

# Query to retrieve few records for test purpose
vic_min = pd.read_sql("(SELECT * FROM victoria_top_min "
                      "WHERE easting > 0 AND northing > 0 ORDER BY utc_date ASC LIMIT 10000) "
                      "UNION ALL"
                      "(SELECT * FROM victoria_top_min "
                      "WHERE easting > 0 AND northing > 0 ORDER BY utc_date DESC LIMIT 10000)", engine)
# Query to retrieve all the dataset
# vic_min = pd.read_sql("SELECT * FROM victoria_top_min WHERE easting > 0 AND northing > 0", engine)

participants = pd.DataFrame({"id": pd.unique(vic_min['inter_id'])})

# # -- Distribution of interested variables
# c_names = ["Dwell-Trip Duration Construction", "Dwell Construction", "Trip Duration Construction",
#           "Trip Length Construction", "Visit Frequency Construction"]
# for c in c_names:
#     f, fb, fa = stable.build_features(vic_min, participants, c)
    # print(c)
    # plot_hist(f, c, c)

f, fb, fa = stable.build_features(vic_min, participants, method="kernel")

# -- StABLE
c = stable.dwell_trip_dur_construction
ngrams = [2, 2]
f, fb, fa, const_name = stable.build_features(vic_min, participants, c, True)
between, within = stable.calc_sims(f, participants, ngrams[0], ngrams[1], fb, fa)


# -- Distribution of similarities
plot_boxplot(between, ngrams[0], ngrams[1], const_name)
plot_bar(within, min_ngram=ngrams[0], max_ngram=ngrams[1], analysis=const_name)
partitions = louvain(between, participants, 0.5, ngrams[0])

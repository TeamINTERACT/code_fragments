import os
from git import Repo
import shutil
import tempfile
import numpy as np
import spatial_metrics as sm
from sklearn.feature_extraction.text import TfidfVectorizer


def download_tripk_repo():
    git_url = "https://git.cs.usask.ca/jrb871/interact-work.git"
    t = tempfile.mkdtemp()
    Repo.clone_from(git_url, t, branch="trip_bugfix")
    shutil.copy(os.path.join(t, 'tools/trip_detection.py'), '.')
    shutil.rmtree(t)


# download_tripk_repo()


import trip_detection as tripk


def simple_trip(df, participants, c_name):
    def dwell_trip_dur_construction(p_movement, n_trip):
        story = ""
        trip_detect = 0
        for index, row in p_movement.iterrows():
            if (row['trip_length'] == 1) & (row['duration'] > n_trip):
                trip_detect += 1

                if trip_detect > 1:
                    story += "t1 "

                story += "d" + str(row['duration']) + " "
            else:
                trip_detect = 0
                story += "t" + str(row["duration"]) + " "

        return story.rstrip()

    def dwell_construction(p_movement, n_trip):
            story = ""
            for index, row in p_movement.iterrows():
                if (row['trip_length'] == 1) & (row['duration'] > n_trip):
                    story += "d" + str(row['duration']) + " "

            return story.rstrip()

    def trip_dur_construction(p_movement, n_trip):
            story = ""
            trip_detect = 0
            for index, row in p_movement.iterrows():
                if (row['trip_length'] == 1) & (row['duration'] > n_trip):
                    trip_detect += 1

                    if trip_detect > 1:
                        story += "t1 "
                else:
                    trip_detect = 0
                    story += "t" + str(row["duration"]) + " "

            return story.rstrip()

    def trip_len_construction(p_movement, n_trip):
            story = ""
            trip_detect = 0
            for index, row in p_movement.iterrows():
                if (row['trip_length'] == 1) & (row['duration'] > n_trip):
                    trip_detect += 1

                    if trip_detect > 1:
                        story += "t1 "
                else:
                    trip_detect = 0
                    story += "t" + str(row["trip_length"]) + " "

            return story.rstrip()

    def visit_frequency_construction(p_movement):
        story = ""
        for grid in p_movement['grid_cell']:
            story += str(grid) + " "

        return story.rstrip()

    func = {"Dwell-Trip Duration Construction": dwell_trip_dur_construction,
                      "Dwell Construction": dwell_construction,
                      "Trip Duration Construction": trip_dur_construction,
                      "Trip Length Construction": trip_len_construction,
                      "Visit Frequency Construction": visit_frequency_construction}

    features = []
    features_before = []
    features_after = []

    n_trip = 3
    df["grid_cell"] = sm.get_grid(df["easting"], df["northing"], 10)
    construction = func[c_name]

    if construction != visit_frequency_construction:
        for p in participants['id']:
            p_data = df[df['inter_id'] == p]
            p_data = sm.movement_detection(p_data, n_trip)

            features.append(construction(p_data, n_trip))

            middle = np.floor(len(p_data)/2).astype(int)
            features_before.append(construction(p_data.iloc[range(0, middle)], n_trip))
            features_after.append(construction(p_data.iloc[range(middle, len(p_data))], n_trip))
    else:
        for p in participants['id']:
            p_data = df[df['inter_id'] == p]

            features.append(construction(p_data))

            middle = np.floor(len(p_data)/2).astype(int)
            features_before.append(construction(p_data.iloc[range(0, middle)]))
            features_after.append(construction(p_data.iloc[range(middle, len(p_data))]))

    return features, features_before, features_after


def kernel_trip(df, participants):
    def bout_visit_construction(bout, visit):
        max_len = len(bout)
        min_len = len(visit)
        max_bout = True

        if len(visit) > max_len:
            max_len = len(visit)
            min_len = len(bout)
            max_bout = False

        story = ""
        for i in range(max_len):
            if i < min_len:
                story += "t" + str(bout.iloc[i]['num_segments']) + " d" + str(visit.iloc[i]['duration']) + " "
            else:
                if max_bout:
                    story += "t" + str(bout.iloc[i]['num_segments']) + " "
                else:
                    story += "d" + str(visit.iloc[i]['duration']) + " "

        return story.rstrip()

    col = {"lat": "lat", "lon": "lon", "utm_n": "northing", "utm_e": "easting"}

    features = []
    features_before = []
    features_after = []

    for p in participants['id']:
        shutil.rmtree(os.getcwd() + "/temp", ignore_errors=True)

        os.makedirs(os.getcwd() + "/temp", exist_ok=True)
        csv_path = os.getcwd() + "/temp/temp_gps_ds5.csv"

        p_data = df[df['inter_id'] == p]
        p_data.to_csv(csv_path, index=False)
        tripk.get_data_gaps(p_data)

        try:
            final_bouts, hotspots, visit_table, gaps = tripk.detect_trips(csv_path, columns=col)
            features.append(bout_visit_construction(final_bouts, visit_table))

            fmiddle = np.floor(len(final_bouts)/2).astype(int)
            vmiddle = np.floor(len(visit_table)/2).astype(int)
            features_before.append(bout_visit_construction(final_bouts.iloc[range(0, fmiddle)],
                                                           visit_table.iloc[range(0, vmiddle)]))
            features_after.append(bout_visit_construction(final_bouts.iloc[range(fmiddle, len(final_bouts))],
                                                          visit_table.iloc[range(vmiddle, len(visit_table))]))
        except Exception:
            features.append(-1)
            features_before.append(-1)
            features_after.append(-1)

    return features, features_before, features_after


def build_features(df, participants, construction=None, method="simple"):
    if method == "simple":
        return simple_trip(df, participants, construction)
    elif method == "kernel":
        return kernel_trip(df, participants)


def calc_sims(f, participants, min_ngram, max_ngram, fb=None, fa=None):
    def cos_sim(features):
        # Method: cosine similarity (similar with repetition) + N-gram (time/sequence) + TF-IDF (normalization)
        vect = TfidfVectorizer(ngram_range=(min_ngram, max_ngram), norm='l2')
        tfidf = vect.fit_transform(features)

        return (tfidf * tfidf.T).A

    # Compare all movements between players
    sims = cos_sim(f)

    sims_within = []
    # Compare before and after movements within players
    if (fb and fa) is not None:
            for i in range(len(participants)):
                if len(fb[i]) > 0 and len(fa[i]) > 0:
                    sims_within.append(cos_sim([fb[i], fa[i]])[0][1])
                else:
                    sims_within.append(-1)

    return sims, sims_within


def filter_sims(sims, participants, threshold):
    similars = []

    for i in range(0, len(sims)):
        for j in range(0, len(sims[i])):
            if j >= i:
                break

            score = sims[i][j]
            if score > threshold:
                similars.append((participants.index[i], participants.index[j], (1 - score)))

    return similars

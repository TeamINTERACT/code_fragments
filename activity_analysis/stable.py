from sys import argv
import interact_tools as it
from os.path import isdir, isfile
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def bout_lengths(df):
    """
    Get sets of consecutive equal activity level values, assign a number to each set, and get the length of each set
    :param df: dataframe containing the individual's activity levels
    :return: dataframe containing each activity bout, including its type and duration
    """
    # Get sets of consecutive equal values, assign a number to each set, and get the length of each set
    df['bout'] = (df.dominant.shift(1) != df.dominant).astype(int).cumsum()
    bouts = df.drop_duplicates(subset=['bout'])[['utcdate', 'dominant', 'bout']].set_index('bout')
    bouts['length'] = df['bout'].value_counts()
    return bouts


def bout_story(bouts):
    """
    Translate a dataframe containing activity bout information into a string describing the participant's
    activity patterns
    :param bouts: dataframe containing activity bout information (output of bout_lengths)
    :return: string describing the participant's activity patterns
    """
    to_ret = ""
    for _, row in bouts.iterrows():
        to_ret = to_ret + row.dominant[0] + str(row.length) + ' '
    return to_ret.rstrip()


def sim_days(df, min_ngram=2, max_ngram=2, subset=None):
    """
    Creates csv files for each participant, which describe how similar that participant's activity patterns are
    between days with a similarity matrix
    :param df: activity bout information dataframe for all participants
    :param subset: the only participant ids we want to consider in df (optional)
    """
    participants = df['interact_id'].drop_duplicates().to_list()
    if subset is not None and all(p in participants for p in subset):
        participants = subset
    for interact_id in participants:
        print(interact_id)
        stories = []
        p_data = df[df.interact_id == interact_id]
        day_list = []
        for day in pd.date_range(p_data.utcdate.min(), p_data.utcdate.max() + timedelta(days=1)):
            if str(day)[0:10] not in p_data.set_index('utcdate').index:
                continue
            durations = bout_lengths(p_data.set_index('utcdate')[str(day)[0:10]].reset_index())
            if len(durations.index) > 2:
                day_list.append(str(day)[0:10])
                story = bout_story(durations)
                stories.append(story)
        vect = TfidfVectorizer(ngram_range=(min_ngram, max_ngram), norm='l2')
        tfidf = vect.fit_transform(stories)
        sim = pd.DataFrame((tfidf * tfidf.T).A, columns=day_list, index=day_list)
        out_fname = output_dir + '/' + str(interact_id) + '_stable_' + it.get_last_commit_date() + '.csv'
        sim.to_csv(out_fname)


def sim_participants(df, min_ngram=2, max_ngram=2, subset=None, out_fname=None):
    """
    Creates a similarity matrix, which describe how similar each participant's activity patterns are
    to each other participant's activity patterns
    :param df: activity bout information dataframe for all participants
    :param subset: the only participant ids we want to consider in df (optional)
    :return: the similarity matrix for comparing participant activity patterns
    """
    participants = df['interact_id'].drop_duplicates().to_list()
    if subset is not None and all(p in participants for p in subset):
        participants = subset
    stories = []
    for interact_id in participants:
        p_data = df[df.interact_id == interact_id]
        stories.append(bout_story(bout_lengths(p_data)))
    vect = TfidfVectorizer(ngram_range=(min_ngram, max_ngram), norm='l2')
    tfidf = vect.fit_transform(stories)
    sim = pd.DataFrame((tfidf * tfidf.T).A, columns=participants, index=participants)
    if out_fname is not None:
        sim.to_csv(out_fname)
    return sim


def sim_pairs(df, subset_1, subset_2, min_ngram=2, max_ngram=2):
    if len(subset_1) != len(subset_2):
        return None
    vect = TfidfVectorizer(ngram_range=(min_ngram, max_ngram), norm='l2')
    for i in range(0, len(subset_1)):
        p_1 = df[df.interact_id == subset_1[i]]
        p_2 = df[df.interact_id == subset_2[i]]
        tfidf = vect.fit_transform([bout_story(bout_lengths(p_1)), bout_story(bout_lengths(p_2))])
        sim = pd.DataFrame((tfidf * tfidf.T).A)
        out_fname = output_dir + '/' + str(subset_1[i]) + '_' + str(subset_2[i]) + '_stable_' + it.get_last_commit_date() + '.csv'
        sim.to_csv(out_fname)


def sim_traits(top_fname, stable_fname, verbose=False, thresh=0.8):
    top = pd.read_csv(top_fname)
    stable = pd.read_csv(stable_fname, index_col=0)
    participants = stable.index.to_list()
    all_similar = {}
    for p in participants:
        sim = stable[(stable.loc[p] > thresh).to_list()].index.to_list()
        sim.remove(p)
        sim.insert(0, p)
        similar = pd.DataFrame()
        for s in sim:
            traits = top[top.interact_id == s].iloc[0][['age', 'gender']]
            traits['similarity'] = stable.loc[p, str(s)]
            similar[s] = traits
        similar = similar.transpose()
        all_similar[p] = similar
        if verbose:
            print("=====================================================")
            print(similar)
    return all_similar


if __name__ == "__main__":
    city, wave = it.get_command_args("stable.py")
    if len(argv) < 4:
        print("Usage: python activity_bouts.py SITE_NAME WAVE_NUMBER DIRECTORY")
        exit()
    output_dir = argv[3]
    if not isdir(output_dir):
        print("Could not locate provided directory.")
        exit()

    ver = it.get_last_commit_date()
    if wave < 10:
        in_fname = output_dir + '/' + city + '_0' + str(wave) + '_min_bouts_' + ver + '.csv'
        stable_fname = output_dir + '/' + city + '_0' + str(wave) + '_stable_' + it.get_last_commit_date() + '.csv'
    else:
        in_fname = output_dir + '/' + city + '_' + str(wave) + '_min_bouts_' + ver + '.csv'
        stable_fname = output_dir + '/' + city + '_' + str(wave) + '_stable_' + it.get_last_commit_date() + '.csv'

    if not isfile(in_fname):
        print(in_fname)
        print("Provided path for file name is not a file.")
        exit()

    pd.options.mode.chained_assignment = None
    bouts_df = pd.read_csv(in_fname)
    bouts_df['utcdate'] = pd.to_datetime(bouts_df['utcdate'])
    if '-o' in argv or not isfile(stable_fname):
        sim_participants(bouts_df, out_fname=stable_fname)
    if '-d' in argv:
        sim_days(bouts_df)
    if '-t' in argv:
        sim_traits(in_fname, stable_fname, True)

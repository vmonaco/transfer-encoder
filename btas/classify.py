import numpy as np
import pandas as pd
from pohmm import Pohmm
from sklearn.metrics import roc_curve


def eer_from_scores(scores):
    far, tpr, thresholds = roc_curve(scores['genuine'], scores['score'], drop_intermediate=False)
    frr = (1 - tpr)
    idx = np.argmin(np.abs(far - frr))

    return np.mean([far[idx], frr[idx]])


def stratified_kfold(df, n_train_samples, n_folds, random=False):
    """
    Create stratified k-folds from an indexed dataframe
    """
    sessions = pd.DataFrame.from_records(list(df.index.unique())).groupby(0).apply(lambda x: x[1].unique())
    indexes = set(df.index.unique())
    folds = []
    for i in range(n_folds):
        train_idx = sessions.apply(lambda x: pd.Series(np.random.choice(x, n_train_samples, replace=False)))
        train_idx = pd.DataFrame(train_idx.stack().reset_index(level=1, drop=True)).set_index(0,
                                                                                              append=True).index.values
        test_idx = list(indexes.difference(train_idx))
        folds.append((df.loc[train_idx], df.loc[test_idx]))

    return folds


def split_dataset(df, template_reps, genuine_reps, impostor_reps):
    df_template = df.groupby(level=0).apply(lambda x: x[x.index.get_level_values(1).isin(template_reps)])
    df_genuine = df.groupby(level=0).apply(lambda x: x[x.index.get_level_values(1).isin(genuine_reps)])
    impostors = df.groupby(level=0).apply(lambda x: x[x.index.get_level_values(1).isin(impostor_reps)]).reset_index(
        level=0, drop=True)

    dfs_impostor = []
    for user in df.index.get_level_values(0).unique():
        df_impostor = impostors.drop(user).reset_index().copy()
        df_impostor['repetition'] = df_impostor['subject'].astype(str) + '_' + df_impostor['repetition'].astype(str)
        df_impostor['subject'] = user
        df_impostor = df_impostor.set_index(['subject', 'repetition'])
        dfs_impostor.append(df_impostor)

    df_impostor = pd.concat(dfs_impostor)

    return df_template, df_genuine, df_impostor


class Classifier(object):
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.models = {}
        return

    def fit(self, X, y):
        """
        Fit the classifier with samples X from identities y
        """
        y = np.array(y)
        unique_y = np.unique(y)
        for yi in unique_y:
            Xi = X[y == yi]
            self.models[yi] = self.model_factory()
            self.models[yi].fit(Xi)

        return

    def score(self, X, y):
        """
        Score the samples X with claimed identities y
        """
        # scores = np.array([self.models[yi].score(Xi) for Xi,yi in zip(X,y)])
        unique_y = np.unique(y)
        scores = np.zeros(len(y))
        for yi in unique_y:
            Xi = X[y == yi]
            scores[y == yi] = self.models[yi].score(Xi)
        return scores


class ScaledManhattan(object):
    def fit(self, X):
        self.X = X
        self.mean = X.mean(axis=0)
        self.absdev = np.abs(X - self.mean).mean(axis=0)
        return

    def score(self, X):
        return - (np.abs(X - self.mean) / self.absdev).sum(axis=1).squeeze()


class Manhattan(object):
    def fit(self, X):
        self.mean = X.mean(axis=0)
        return

    def score(self, X):
        return - (np.abs(X - self.mean)).sum(axis=1).squeeze()/len(self.mean)


class POHMM(object):
    def __init__(self):
        self.pohmm = Pohmm(n_hidden_states=2,
                           init_spread=2,
                           emissions=['lognormal', 'lognormal'],
                           smoothing='freq',
                           init_method='obs',
                           thresh=1e-2)

    def fit(self, X):
        X = np.c_[np.median(X[:, 1::2], axis=1), X].reshape(-1, 17, 2)
        pstates = np.repeat([np.arange(17)], len(X), axis=0)
        return self.pohmm.fit(X, pstates)

    def score(self, X):
        X = np.c_[np.median(X[:, 1::2], axis=1), X].reshape(-1, 17, 2)
        pstates = np.repeat([np.arange(17)], len(X), axis=0)
        scores = np.zeros(len(X))
        for i, (Xi, pstates_i) in enumerate(zip(X, pstates)):
            scores[i] = self.pohmm.score(Xi, pstates_i)
        return scores


def verification_scores(cl, df_template, df_genuine, df_impostor, normalize_scores=False):
    def extract_labels_samples(df):
        return df.index.get_level_values(0).values, df.values

    if df_template is not None:
        # Train the classifier
        template_labels, template_samples = extract_labels_samples(df_template)
        cl.fit(template_samples, template_labels)

    genuine_labels, genuine_samples = extract_labels_samples(df_genuine)
    impostor_labels, impostor_samples = extract_labels_samples(df_impostor)

    # Genuine and impostor scores
    genuine_scores = cl.score(genuine_samples, genuine_labels)
    impostor_scores = cl.score(impostor_samples, impostor_labels)

    users = np.r_[genuine_labels, impostor_labels]
    genuine = np.array([True] * len(genuine_scores) + [False] * len(impostor_scores))
    scores = np.r_[genuine_scores, impostor_scores]

    if normalize_scores:
        for user in np.unique(users):
            user_idx = users == user

            # Normalize scores between min and max for each claimed user (each model)
            min_score, max_score = scores[user_idx].min(), scores[user_idx].max()

            scores[user_idx] = (scores[user_idx] - min_score) / (max_score - min_score)

        scores[scores < 0] = 0
        scores[scores > 1] = 1

    results = pd.DataFrame(list(zip(users, genuine, scores)), columns=['subject', 'genuine', 'score'])

    # Summary of the EER
    summary = results.groupby('subject').apply(eer_from_scores).describe()

    return results, summary

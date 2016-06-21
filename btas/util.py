import os
import tempfile
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from urllib import request

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
FIGURES_DIR = os.path.join(ROOT_DIR, 'figures')

CMU_URL = 'http://www.cs.cmu.edu/~keystroke/'
CMU_FNAME = 'DSL-StrongPasswordData.csv'

BIOSIG_URL = 'http://www.epaymentbiometrics.ensicaen.fr/wp-content/uploads/2015/04/'
BIOSIG_FNAME = 'greyc-nislab-keystroke-benchmark-dataset.xls'


def maybe_mkdir(dirpath=None):
    if dirpath is None:
        dirpath = tempfile.mkdtemp()
    elif not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return dirpath


def maybe_download(source_url, filename, destdir):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(destdir):
        os.makedirs(destdir, mode=0o755)
    filepath = os.path.join(destdir, filename)
    if not os.path.exists(filepath):
        filepath, _ = request.urlretrieve(source_url + filename, filepath)
        with open(filepath) as f:
            size = f.seek(0, 2)
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def load_cmu():
    fname = maybe_download(CMU_URL, CMU_FNAME, DATA_DIR)
    df = pd.read_csv(fname, index_col=[0, 1, 2])
    repetition = (df.index.get_level_values(1) - 1) * 50 + df.index.get_level_values(2)
    df = df.reset_index(level=[1, 2], drop=True)
    df = df.set_index(repetition, append=True)
    df.index.names = ['subject', 'repetition']
    return df


def load_biosig(i, h=2):
    fname = maybe_download(BIOSIG_URL, BIOSIG_FNAME, DATA_DIR)
    df = pd.read_excel(fname, sheetname=i)

    df = df[df['Class'] == h]

    df['repetition'] = 0
    df['subject'] = df['User_ID']

    def make_repetitions(x):
        x['repetition'] = np.arange(1, len(x) + 1)
        return x

    df = df.groupby(['subject', 'repetition']).apply(make_repetitions)

    def make_features(x):
        return pd.Series(x['Keystroke Template Vector'].split()).astype(int) / 1e7

    features = df.apply(make_features, axis=1)

    features['subject'] = df['subject']
    features['repetition'] = df['repetition']

    features = features.set_index(['subject', 'repetition'])

    return features


def save_fig(name, ext='pdf'):
    plt.savefig(os.path.join(FIGURES_DIR, name + '.%s' % ext), bbox_inches='tight')
    plt.close()
    return


def save_results(df, name):
    df.to_csv(os.path.join(RESULTS_DIR, name + '.csv'))
    return


def load_results(name, **args):
    return pd.read_csv(os.path.join(RESULTS_DIR, name + '.csv'), **args)

    import os
import numpy as np
import pandas as pd
from itertools import cycle, product
from sklearn.metrics import roc_curve
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from btas.classify import split_dataset, Classifier, Manhattan, ScaledManhattan, verification_scores, eer_from_scores
from btas.util import load_cmu, load_biosig, save_results, save_fig, load_results
from btas.transfer import TransferEncoder

FIGSIZE = (6, 4)
HIST_UPPER = 1

# Seeds each experiment, primarily random nums for initial network weights
SEED = 1234

# Number of random splits in each scenario
N_VALIDATIONS = 10

# Number of users to train the transfer encoder
N_CMU_TRANSFER_USERS = 25
N_BIOSIG_TRANSFER_USERS = 84

# Biosig dataset has 5 different input types (one per Excel sheet)
N_BIOSIG_PASSWORDS = 5

# Use a Manhattan distance anomaly detector, similar to #8 in Maxion's work
CL_FACTORY = lambda: Classifier(lambda: Manhattan())

# Functions to generate transfer encoders
NONE_TE_FACTORY = lambda: None
CMU_TE_FACTORY = lambda: TransferEncoder(n_hidden=300, n_steps=1000)
BIOSIG_TE_FACTORY = lambda: TransferEncoder(n_hidden=1000, n_steps=1000)

# Each scenario consists of: template reps, Genuine reps, impostor reps
CMU_MOTIVATION_SCENARIOS = [
    (np.array([1]), np.arange(2, 201), np.arange(1, 6)),
    (np.array([1]), np.arange(2, 201), np.arange(396, 401)),
    (np.array([1]), np.arange(201, 401), np.arange(1, 6)),
    (np.array([1]), np.arange(201, 401), np.arange(396, 401)),

    # Maxion scenario
    (np.arange(1, 201), np.arange(201, 401), np.arange(1, 6)),

    # Maxion scenario with practiced impostors
    (np.arange(1, 201), np.arange(201, 401), np.arange(396, 401)),
]

# Each scenario consists of: template reps, Genuine reps, impostor reps
CMU_TRANSFER_SCENARIOS = [
    (np.array([1]), np.arange(2, 201), np.arange(1, 11)),
    (np.array([1]), np.arange(2, 201), np.arange(391, 401)),
    (np.array([1]), np.arange(201, 401), np.arange(1, 11)),
    (np.array([1]), np.arange(201, 401), np.arange(391, 401)),

    # Maxion scenario
    (np.arange(1, 201), np.arange(201, 401), np.arange(1, 11)),

    # Maxion scenario with practiced impostors
    (np.arange(1, 201), np.arange(201, 401), np.arange(391, 401)),
]

# The transfer encoder learns to encode all the source reps as target reps
CMU_SOURCE_REPS = np.arange(1, 51)
CMU_TARGET_REPS = np.arange(351, 401)

# Number of hands for hands/reps for (template, genuine, impostor)
BIOSIG_TRANSFER_SCENARIOS = [
    (1, 1, 1, np.array([1]), np.arange(1, 11), np.array([1])),
    (1, 1, 2, np.array([1]), np.arange(1, 11), np.array([1])),
    (1, 2, 1, np.array([1]), np.arange(2, 11), np.array([1])),
    (1, 2, 2, np.array([1]), np.arange(2, 11), np.array([1])),
]

SCENARIO_LETTER = ['A', 'B', 'C', 'D', 'B/C', 'A/D']


def normalize(df):
    lower = df.min(axis=0)
    upper = df.max(axis=0)
    return (df - lower) / (upper - lower)


def cmu_features_vs_repetitions(n_users=2, window=100, seed=SEED):
    np.random.seed(seed)
    df = load_cmu()  # Not normalized

    users = np.random.choice(df.index.levels[0].unique(), n_users)

    linestyles = ['-', '--', '-.', ':']

    # First two hold and DD latencies
    features = ['H.period', 'DD.period.t', 'H.t', 'DD.t.i']

    df = df[features]

    fig, ax = plt.subplots(n_users, 2, figsize=(4, 4), sharex=True, sharey=True)
    for user_i, user in enumerate(users):
        df_user = df.loc[user]
        for style, feature in zip(cycle(linestyles), features):
            ax[user_i, 0].plot(np.r_[df_user[:window][feature].expanding().mean(),
                                     df_user[feature].rolling(window=window).mean().dropna().values],
                               label=feature, linestyle=style, linewidth=1)
            ax[user_i, 1].plot(np.r_[df_user[:window][feature].expanding().std(),
                                     df_user[feature].rolling(window=window).std().dropna().values],
                               label=feature, linestyle=style, linewidth=1)

            ax[user_i, 1].set_xticks(np.linspace(0, 400, 5))

    ax[0, 0].set_title('Rolling mean')
    ax[0, 1].set_title('Rolling SD')

    ax[0, 1].legend()
    fig.tight_layout()

    fig.text(0.5, 0.0, 'Repetition', ha='center')
    # fig.text(0.0, 0.5, 'Feature', va='center', rotation='vertical')
    fig.text(0.0, 0.25, 'User B', va='center', rotation='vertical')
    fig.text(0.0, 0.75, 'User A', va='center', rotation='vertical')

    # plt.show()
    save_fig('features_vs_repetitions')
    return


def cmu_motivation_results(cl_factory, seed=SEED):
    np.random.seed(seed)
    df = load_cmu()  # Not normalized

    results = []
    for template_reps, genuine_reps, impostor_reps in CMU_MOTIVATION_SCENARIOS:
        df_template, df_genuine, df_impostor = split_dataset(df,
                                                             template_reps=template_reps,
                                                             genuine_reps=genuine_reps,
                                                             impostor_reps=impostor_reps)
        cl = cl_factory()
        classifier_scores, classifier_summary = verification_scores(cl, df_template, df_genuine, df_impostor)
        eer = '%.3f (%.3f)' % (classifier_summary['mean'], classifier_summary['std'])

        row = (
            '[%d:%d]' % (template_reps[0], template_reps[-1]),
            '[%d:%d]' % (genuine_reps[0], genuine_reps[-1]),
            '[%d:%d]' % (impostor_reps[0], impostor_reps[-1]),
            eer
        )
        results.append(row)

    results = pd.DataFrame(results, columns=['Template', 'Genuine', 'Impostor', 'EER'])
    save_results(results, 'cmu_motivation')
    print(results)
    return


def fit_te(te, df_source, df_target, one_to_one, shuffle_users):
    inputs, outputs = [], []
    if shuffle_users:
        u1 = df_source.index.get_level_values(0).unique()
        u2 = np.random.permutation(u1)
        while np.any(u1 == u2):
            u2 = np.random.permutation(u1)
        user_pairs = list(zip(u1, u2))
    else:
        user_pairs = [(u, u) for u in df_source.index.get_level_values(0).unique()]

    if one_to_one:
        iter_fun = zip
    else:
        iter_fun = product

    for u1, u2 in user_pairs:
        idx1 = np.arange(df_source.loc[u1].shape[0])
        idx2 = np.arange(df_target.loc[u2].shape[0])
        idx = np.array(list(iter_fun(idx1, idx2)))
        inputs.append(df_source.loc[u1].values[idx[:, 0]])
        outputs.append(df_target.loc[u2].values[idx[:, 1]])

    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)

    te.fit(inputs, outputs)
    return te


def cmu_results(name, te_factory, cl_factory,
                one_to_one=True,
                shuffle_users=False,
                n_transfer_users=N_CMU_TRANSFER_USERS,
                n_validations=N_VALIDATIONS, seed=SEED):
    np.random.seed(seed)
    df = load_cmu()
    df = normalize(df)

    score_dfs = []
    for validation_idx in range(n_validations):
        source_users = list(np.random.choice(df.index.get_level_values(0).unique(), n_transfer_users, replace=False))
        target_users = list(df.index.levels[0].difference(source_users))
        source = df.loc[source_users]
        target = df.loc[target_users]

        te = te_factory()
        if te:
            df_source = source.groupby(level=0).apply(
                lambda x: x[x.index.get_level_values(1).isin(CMU_SOURCE_REPS)]).reset_index(level=0, drop=True)
            df_target = source.groupby(level=0).apply(
                lambda x: x[x.index.get_level_values(1).isin(CMU_TARGET_REPS)]).reset_index(level=0, drop=True)

            te = fit_te(te, df_source, df_target, one_to_one, shuffle_users)

        for scenario_idx, (template_reps, genuine_reps, impostor_reps) in enumerate(CMU_TRANSFER_SCENARIOS):
            df_template, df_genuine, df_impostor = split_dataset(target.copy(),
                                                                 template_reps=template_reps,
                                                                 genuine_reps=genuine_reps,
                                                                 impostor_reps=impostor_reps)
            if te:
                # Transfer the template data to the target domain
                df_template.values[:] = te.transfer(df_template.values)

            cl = cl_factory()
            classifier_scores, _ = verification_scores(cl, df_template, df_genuine,
                                                       df_impostor)
            classifier_scores['scenario'] = scenario_idx
            classifier_scores['fold'] = validation_idx

            score_dfs.append(classifier_scores)

    scores = pd.concat(score_dfs).set_index(['scenario', 'fold', 'subject'])
    score_summarys = scores.groupby(level=['scenario', 'fold', 'subject']).apply(eer_from_scores).groupby(
        level=['scenario']).describe()

    summary_rows = []
    for scenario_idx, (template_reps, genuine_reps, impostor_reps) in enumerate(CMU_TRANSFER_SCENARIOS):
        scenario_eer = score_summarys.loc[scenario_idx]

        eer = '%.3f (%.3f)' % (scenario_eer['mean'], scenario_eer['std'])

        row = (
            '[%d:%d]' % (template_reps[0], template_reps[-1]),
            '[%d:%d]' % (genuine_reps[0], genuine_reps[-1]),
            '[%d:%d]' % (impostor_reps[0], impostor_reps[-1]),
            eer
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows, columns=['Template', 'Genuine', 'Impostor', 'EER'])

    print(name)
    print(summary)

    save_results(summary, 'cmu_summary_%s' % name)
    save_results(scores, 'cmu_scores_%s' % name)

    # Plot the score distributions
    fig, axs = plt.subplots(len(CMU_TRANSFER_SCENARIOS), 1, figsize=(4, 8), sharex=True, sharey=True)
    for scenario_idx, ax in enumerate(axs):
        s = scores.loc[scenario_idx]
        s.set_index('genuine')
        genuine = s.set_index('genuine').loc[True]['score']
        impostor = s.set_index('genuine').loc[False]['score']
        sns.distplot(-genuine.values, color='blue', bins=np.linspace(0, HIST_UPPER, 31), label='Genuine',
                     norm_hist=True,
                     ax=ax, axlabel=False)
        sns.distplot(-impostor.values, color='red', bins=np.linspace(0, HIST_UPPER, 31), label='Impostor',
                     norm_hist=True,
                     ax=ax, axlabel=False)
        ax.set_xlim(0, HIST_UPPER)
        ax.text(0.5, 0.9, SCENARIO_LETTER[scenario_idx], ha='center', va='center', transform=ax.transAxes)

    axs[0].legend(loc='upper right')
    plt.tight_layout()
    save_fig('cmu_%s_score_distributions' % name)

    for scenario_idx in range(len(CMU_TRANSFER_SCENARIOS)):
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, sharex=True, sharey=True)
        s = scores.loc[scenario_idx]
        s.set_index('genuine')
        genuine = s.set_index('genuine').loc[True]['score']
        impostor = s.set_index('genuine').loc[False]['score']
        sns.distplot(-genuine.values, color='blue', bins=np.linspace(0, HIST_UPPER, 31), label='Genuine',
                     norm_hist=True,
                     ax=ax, axlabel=False)
        sns.distplot(-impostor.values, color='red', bins=np.linspace(0, HIST_UPPER, 31), label='Impostor',
                     norm_hist=True,
                     ax=ax, axlabel=False)
        ax.set_xlim(0, HIST_UPPER)
        ax.legend(loc='upper right')
        plt.tight_layout()
        save_fig('cmu_%s_scenario_%s_score_distributions' % (name, SCENARIO_LETTER[scenario_idx].replace('/', '-')))

    return


def biosig_motivation_results(cl_factory, seed=SEED):
    np.random.seed(seed)

    score_dfs = []
    for p in range(N_BIOSIG_PASSWORDS):
        h1 = load_biosig(p, 1)
        h2 = load_biosig(p, 2)
        for scenario_idx, (
                template_hands, genuine_hands, impostor_hands, template_reps, genuine_reps,
                impsotor_reps) in enumerate(
            BIOSIG_TRANSFER_SCENARIOS):
            df_template, df_genuine, df_impostor = biosig_split_data(h1.copy(),
                                                                     h2.copy(),
                                                                     template_hands, genuine_hands, impostor_hands,
                                                                     template_reps=template_reps,
                                                                     genuine_reps=genuine_reps,
                                                                     impostor_reps=impsotor_reps)

            cl = cl_factory()
            scores, _ = verification_scores(cl, df_template, df_genuine, df_impostor)

            scores['password'] = p + 1
            scores['scenario'] = scenario_idx

            score_dfs.append(scores)

    scores = pd.concat(score_dfs).set_index(['scenario', 'password', 'subject'])
    score_summarys = scores.groupby(level=['scenario', 'password', 'subject']).apply(eer_from_scores).groupby(
        level=['scenario']).describe()

    summary_rows = []
    for scenario_idx, (
            template_hands, genuine_hands, impostor_hands, template_reps, genuine_reps, impsotor_reps) in enumerate(
        BIOSIG_TRANSFER_SCENARIOS):
        scenario_eer = score_summarys.loc[scenario_idx]

        eer = '%.3f (%.3f)' % (scenario_eer['mean'], scenario_eer['std'])

        row = (
            '%d' % template_hands,
            '%d' % genuine_hands,
            '%d' % impostor_hands,
            eer
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows, columns=['Template', 'Genuine', 'Impostor', 'EER'])
    print(summary)
    save_results(summary, 'biosig_motivation')
    return


def biosig_split_data(h1, h2, template_hands, genuine_hands, impostor_hands, template_reps, genuine_reps,
                      impostor_reps):
    df_template_1, df_genuine_1, df_impostor_1 = split_dataset(h1, template_reps, genuine_reps, impostor_reps)
    df_template_2, df_genuine_2, df_impostor_2 = split_dataset(h2, template_reps, genuine_reps, impostor_reps)

    if template_hands == 1:
        df_template = df_template_1
    else:
        df_template = df_template_2

    if genuine_hands == 1:
        df_genuine = df_genuine_1
    else:
        df_genuine = df_genuine_2

    if impostor_hands == 1:
        df_impostor = df_impostor_1
    else:
        df_impostor = df_impostor_2

    return df_template, df_genuine, df_impostor


def biosig_results(name, te_factory, cl_factory,
                   one_to_one=True,
                   shuffle_users=False,
                   n_transfer_users=N_BIOSIG_TRANSFER_USERS,
                   n_validations=N_VALIDATIONS, seed=SEED):
    np.random.seed(seed)

    score_dfs = []
    for p in range(N_BIOSIG_PASSWORDS):
        h1 = normalize(load_biosig(p, 1))
        h2 = normalize(load_biosig(p, 2))

        for validation_idx in range(n_validations):
            transfer_users = list(
                np.random.choice(h1.index.get_level_values(0).unique(), n_transfer_users, replace=False))
            validation_users = list(h1.index.levels[0].difference(transfer_users))

            te = te_factory()
            if te:
                # Go from one hand to two hands
                df_source = h1.loc[transfer_users]
                df_target = h2.loc[transfer_users]
                te = fit_te(te, df_source, df_target, one_to_one, shuffle_users)

            for scenario_idx, (
                    template_hands, genuine_hands, impostor_hands, template_reps, genuine_reps,
                    impsotor_reps) in enumerate(
                BIOSIG_TRANSFER_SCENARIOS):
                df_template, df_genuine, df_impostor = biosig_split_data(h1.loc[validation_users].copy(),
                                                                         h2.loc[validation_users].copy(),
                                                                         template_hands, genuine_hands, impostor_hands,
                                                                         template_reps=template_reps,
                                                                         genuine_reps=genuine_reps,
                                                                         impostor_reps=impsotor_reps)

                if te:
                    df_template = pd.DataFrame(te.transfer(df_template.values), index=df_template.index)

                cl = cl_factory()
                scores, _ = verification_scores(cl, df_template, df_genuine, df_impostor)

                scores['password'] = p + 1
                scores['scenario'] = scenario_idx
                scores['fold'] = validation_idx

                score_dfs.append(scores)

    scores = pd.concat(score_dfs).set_index(['scenario', 'password', 'fold', 'subject'])
    score_summarys = scores.groupby(level=['scenario', 'password', 'fold', 'subject']).apply(eer_from_scores).groupby(
        level=['scenario']).describe()

    summary_rows = []
    for scenario_idx, (
            template_hands, genuine_hands, impostor_hands, template_reps, genuine_reps, impsotor_reps) in enumerate(
        BIOSIG_TRANSFER_SCENARIOS):
        scenario_eer = score_summarys.loc[scenario_idx]

        eer = '%.3f (%.3f)' % (scenario_eer['mean'], scenario_eer['std'])

        row = (
            '%d' % template_hands,
            '%d' % genuine_hands,
            '%d' % impostor_hands,
            eer
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows, columns=['Template', 'Genuine', 'Impostor', 'EER'])

    print(name)
    print(summary)

    save_results(summary, 'biosig_summary_%s' % name)
    save_results(scores, 'biosig_scores_%s' % name)

    fig, axs = plt.subplots(len(BIOSIG_TRANSFER_SCENARIOS), 1, figsize=(4, 8), sharex=True, sharey=True)
    for scenario_idx, ax in enumerate(axs):
        s = scores.loc[scenario_idx]
        s.set_index('genuine')
        genuine = s.set_index('genuine').loc[True]['score']
        impostor = s.set_index('genuine').loc[False]['score']
        sns.distplot(-genuine.values, color='blue', bins=np.linspace(0, HIST_UPPER, 31), label='Genuine',
                     norm_hist=True,
                     ax=ax, axlabel=False)
        sns.distplot(-impostor.values, color='red', bins=np.linspace(0, HIST_UPPER, 31), label='Impostor',
                     norm_hist=True,
                     ax=ax, axlabel=False)
        ax.set_xlim(0, HIST_UPPER)
        ax.text(0.5, 0.9, SCENARIO_LETTER[scenario_idx], ha='center', va='center', transform=ax.transAxes)

    axs[0].legend(loc='upper right')
    plt.tight_layout()
    save_fig('biosig_%s_score_distributions' % name)

    for scenario_idx in range(len(BIOSIG_TRANSFER_SCENARIOS)):
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, sharex=True, sharey=True)
        s = scores.loc[scenario_idx]
        s.set_index('genuine')
        genuine = s.set_index('genuine').loc[True]['score']
        impostor = s.set_index('genuine').loc[False]['score']
        sns.distplot(-genuine.values, color='blue', bins=np.linspace(0, HIST_UPPER, 31), label='Genuine',
                     norm_hist=True,
                     ax=ax, axlabel=False)
        sns.distplot(-impostor.values, color='red', bins=np.linspace(0, HIST_UPPER, 31), label='Impostor',
                     norm_hist=True,
                     ax=ax, axlabel=False)
        ax.set_xlim(0, HIST_UPPER)
        ax.legend(loc='upper right')
        plt.tight_layout()
        save_fig('biosig_%s_scenario_%s_score_distributions' % (name, SCENARIO_LETTER[scenario_idx]))

    return


def cmu_user_score_figure(scenario=3, user='s003'):
    df1 = load_results('cmu_scores_no_transfer', index_col=[0, 1, 2, 3])
    df2 = load_results('cmu_scores_transfer_N-N', index_col=[0, 1, 2, 3])

    fig, ax = plt.subplots(2, 1, figsize=(5, 3.2), sharex=True, sharey=True)

    sns.distplot(-31 * df1.loc[scenario, :, user, True].values.flatten(), norm_hist=True, color='blue', ax=ax[0],
                 label='Genuine')
    sns.distplot(-31 * df1.loc[scenario, :, user, False].values.flatten(), norm_hist=True, color='red', ax=ax[0],
                 label='Impostor')
    sns.distplot(-31 * df2.loc[scenario, :, user, True].values.flatten(), norm_hist=True, color='blue', ax=ax[1],
                 label='Genuine')
    sns.distplot(-31 * df2.loc[scenario, :, user, False].values.flatten(), norm_hist=True, color='red', ax=ax[1],
                 label='Impostor')

    ax[0].text(0.5, 0.9, 'No transfer', ha='center', va='center', transform=ax[0].transAxes)
    ax[1].text(0.5, 0.9, 'N:N transfer', ha='center', va='center', transform=ax[1].transAxes)

    ax[1].set_xlabel('Manhattan Distance')
    fig.text(0.0, 0.5, 'Density', ha='center', va='center', rotation='vertical')

    ax[0].legend(loc='upper right')
    plt.tight_layout()

    save_fig('cmu_user_score_dists')
    return


def interp_roc(s, far_interp=np.linspace(0, 1, 101)):
    far, tpr, thresholds = roc_curve(s['genuine'], s['score'], drop_intermediate=True)
    frr = (1 - tpr)
    df = pd.DataFrame({'FAR': far_interp, 'FRR': np.interp(far_interp, far, frr)})
    return df

def roc_figure(names, labels, scenario, output=None):

    rocs = []
    for name, label in zip(names, labels):
        scores = load_results(name).set_index(['scenario', 'fold', 'subject']).loc[scenario, :, :]

        roc = scores.groupby(level=[0, 1]).apply(interp_roc)
        roc['Strategy'] = label
        roc['fold_subject'] = roc.index.get_level_values(0).astype(str).values + '_' + roc.index.get_level_values(
            1).astype(str).values
        rocs.append(roc)

    roc = pd.concat(rocs)

    plt.figure(figsize=(4, 4))
    sns.tsplot(roc, time='FAR', value='FRR', condition='Strategy', unit='fold_subject', linewidth=1.0)

    plt.legend(title='Bipartite strategy', loc='upper right')

    plt.xlabel('False acceptance rate')
    plt.ylabel('False rejection rate')

    if output:
        save_fig(output)
    else:
        plt.show()

    return

def roc_2column_figure(datasets, names, labels, scenario, output=None):

    fig, axes = plt.subplots(1, 2, figsize=(5.4,2.5), sharey=True, sharex=True)

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        if i == 0:
            rocs = []
            for name, label in zip(names, labels):
                scores = load_results(dataset + '_' + name).set_index(['scenario', 'fold', 'subject']).loc[scenario, :, :]

                roc = scores.groupby(level=[0, 1]).apply(interp_roc)
                roc['Strategy'] = label
                roc['fold_subject'] = roc.index.get_level_values(0).astype(str).values + '_' + roc.index.get_level_values(
                    1).astype(str).values
                rocs.append(roc)

            roc = pd.concat(rocs)
        else:
            rocs = []
            for name, label in zip(names, labels):
                scores = load_results(dataset + '_' + name).set_index(['scenario', 'fold', 'subject','password']).loc[scenario, :,
                         :,:]

                roc = scores.groupby(level=[0, 1, 2]).apply(interp_roc)
                roc['Strategy'] = label
                roc['fold_subject'] = roc.index.get_level_values(0).astype(
                    str).values + '_' + roc.index.get_level_values(
                    1).astype(str).values + '_' + roc.index.get_level_values(
                    2).astype(str).values
                rocs.append(roc)

            roc = pd.concat(rocs)

        if i == 1:
            legend = True
        else:
            legend = False

        sns.tsplot(roc, time='FAR', value='FRR', condition='Strategy', unit='fold_subject', linewidth=1.0, ax=ax, legend=legend, ci=95)

        ax.lines[0].set_linestyle('--')

        if i == 0:
            ax.set_ylabel('False rejection rate')
            ax.text(0.5, 0.95, 'Low -> high',
                       horizontalalignment='center',
                       verticalalignment='top',
                       transform=ax.transAxes)
        else:
            ax.set_ylabel('')
            plt.legend(title='Bipartite strategy', loc='upper right')
            ax.text(0.5, 0.95, 'One -> both',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)

            ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0),
                      fancybox=True, shadow=True, ncol=5)

        ax.set_xlabel('False acceptance rate')
        # plt.ylabel('False rejection rate')

    plt.tight_layout()

    if output:
        save_fig(output)
    else:
        plt.show()

    return

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # cmu_motivation_results(cl_factory=CL_FACTORY)
    # biosig_motivation_results(cl_factory=CL_FACTORY)
    #
    # cmu_results('no_transfer',
    #             te_factory=NONE_TE_FACTORY,
    #             cl_factory=CL_FACTORY)
    # cmu_results('transfer_1-1',
    #             one_to_one=True,
    #             shuffle_users=False,
    #             te_factory=CMU_TE_FACTORY,
    #             cl_factory=CL_FACTORY)
    # cmu_results('transfer_N-N',
    #             one_to_one=False,
    #             shuffle_users=False,
    #             te_factory=CMU_TE_FACTORY,
    #             cl_factory=CL_FACTORY)
    # cmu_results('transfer_shuffled_1-1',
    #             one_to_one=True,
    #             shuffle_users=True,
    #             te_factory=CMU_TE_FACTORY,
    #             cl_factory=CL_FACTORY)
    # cmu_results('transfer_shuffled_N-N',
    #             one_to_one=False,
    #             shuffle_users=True,
    #             te_factory=CMU_TE_FACTORY,
    #             cl_factory=CL_FACTORY)
    #
    # biosig_results('no_transfer',
    #                te_factory=NONE_TE_FACTORY,
    #                cl_factory=CL_FACTORY)
    # biosig_results('transfer_1-1',
    #                one_to_one=True,
    #                shuffle_users=False,
    #                te_factory=BIOSIG_TE_FACTORY,
    #                cl_factory=CL_FACTORY)
    # biosig_results('transfer_N-N',
    #                one_to_one=False,
    #                shuffle_users=False,
    #                te_factory=BIOSIG_TE_FACTORY,
    #                cl_factory=CL_FACTORY)
    # biosig_results('transfer_shuffled_1-1',
    #                one_to_one=True,
    #                shuffle_users=True,
    #                te_factory=BIOSIG_TE_FACTORY,
    #                cl_factory=CL_FACTORY)
    # biosig_results('transfer_shuffled_N-N',
    #                one_to_one=False,
    #                shuffle_users=True,
    #                te_factory=BIOSIG_TE_FACTORY,
    #                cl_factory=CL_FACTORY)
    #
    # cmu_user_score_figure()
    # cmu_features_vs_repetitions()
    #
    # roc_figure(['cmu_scores_no_transfer',
    #                 'cmu_scores_transfer_1-1',
    #                 'cmu_scores_transfer_N-N',
    #                 'cmu_scores_transfer_shuffled_1-1',
    #                 'cmu_scores_transfer_shuffled_N-N'],
    #                ['None',
    #                 '1:1',
    #                 'N:N',
    #                 '1x1',
    #                 'NxN'],
    #                scenario=2,
    #                output='cmu_roc')
    #
    # roc_figure(['biosig_scores_no_transfer',
    #                    'biosig_scores_transfer_1-1',
    #                    'biosig_scores_transfer_N-N',
    #                    'biosig_scores_transfer_shuffled_1-1',
    #                    'biosig_scores_transfer_shuffled_N-N'],
    #                   ['None',
    #                    '1:1',
    #                    'N:N',
    #                    '1x1',
    #                    'NxN'],
    #                   scenario=2,
    #                   output='biosig_roc')

    roc_2column_figure(['cmu', 'biosig'],
                       ['scores_no_transfer',
                        'scores_transfer_1-1',
                        'scores_transfer_N-N',
                        'scores_transfer_shuffled_1-1',
                        'scores_transfer_shuffled_N-N'],
                       ['None',
                        '1:1',
                        'N:N',
                        '1x1',
                        'NxN'],
                       scenario=2,
                       output='scenario_c_roc')

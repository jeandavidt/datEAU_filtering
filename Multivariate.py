import numpy as np
import numpy.linalg as alg
import pandas as pd

from scipy.stats import f as fisher
from scipy.stats import norm as normal
from sklearn.preprocessing import StandardScaler

import PlottingTools

def build_pca(x_bar, min_var_exp):
    # singular value decomposition
    svd_feat = {}
    svd_feat['u'], svd_feat['s'], p_T = alg.svd(x_bar, full_matrices=False)
    svd_feat['p'] = p_T.T
    svd_feat['cov'] = np.dot(x_bar.T, x_bar) / (len(x_bar) - 1)
    svd_feat['eigenvalues'] = svd_feat['s'] ** 2 / (len(x_bar) - 1)
    svd_feat['Lambda'] = np.diag(svd_feat['eigenvalues'])
    svd_feat['t'] = np.dot(svd_feat['u'], np.diag(svd_feat['s']))

    # How much of the variance is explained by the principal components?
    t_var = []
    n_components = x_bar.shape[1]

    for i in range(n_components):
        t_var.append(svd_feat['t'][:, i].var())

    sum_var = np.sum(t_var)
    svd_feat['exp_var'] = np.divide(t_var, sum_var)

    # Dimension reduction
    # Initializing a model object (dictionnary)
    model = {}

    # What proportion of the variance is explained by the model? Should be over min_var_exp
    explained = 0.
    for i in range(n_components):
        explained += svd_feat['exp_var'][i]
        if (i >= 1 and explained > min_var_exp):
            model['n_comp'] = i + 1
            model['exp_var'] = explained
            break
        else:
            continue
    if model['exp_var'] == 1:
        print('''No dimension reduction has occurred (minimum explained variance too high.
        Fault detection will thus not use the Q test, as all the variance is explained by the model.''')

    # Splitting the matrices
    t_hat = svd_feat['t'][:, 0: model['n_comp']]
    model['t_hat_stdev'] = [t_hat[:, i].std() for i in range(model['n_comp'])]
    model['loading_mat'] = svd_feat['p'][:, 0: model['n_comp']]
    model['lambda_hat'] = svd_feat['Lambda'][0:model['n_comp'], 0:model['n_comp']]

    return model, svd_feat

def calc_stats(norm_data, model_load, eigenvals_mat):
    # Calculate the limits for fault detection
    n_comp = norm_data.shape[1]
    T2 = []
    Q = []
    for i in range(len(norm_data)):
        T2.append(
            np.dot(
                norm_data[i].T,
                np.dot(
                    model_load,
                    np.dot(
                        alg.inv(eigenvals_mat),
                        np.dot(
                            model_load.T,
                            norm_data[i]
                        )
                    )
                )
            )
        )

        R = np.identity(n_comp) - np.dot(model_load, model_load.T)
        Q.append(
            np.dot(
                np.dot(
                    norm_data[i],
                    R
                ),
                norm_data[i].T
            )
        )

    Q = np.array(Q)
    T2 = np.array(T2)
    return T2, Q

def calc_limits(data, n_pc, alpha, eigenvals_vec):
    # calculating the limits
    # T2 stat limit
    limits = {}
    limits['alpha'] = alpha
    n_data = len(data)
    n_vars = data.shape[1]
    F_alpha = fisher.interval(alpha, n_pc, n_data - n_pc)[1]
    T2_lim = (n_pc * (n_data ** 2 - 1) / (n_data * (n_data - n_pc))) * F_alpha
    limits['T2'] = T2_lim
    # Q stat limit prior info
    theta = np.zeros([3, ])
    if n_vars == n_pc:
        Q_lim = np.inf
    else:
        for i in range(1, 4):
            for j in range(n_pc - 1, n_vars):
                theta[i - 1] += eigenvals_vec[j] ** (i)

        h_0 = 1 - ((2 * theta[0] * theta[2]) / (3 * theta[1] ** 2))
        z_alpha = normal.ppf(alpha)

        # Actual Q stat limit
        Q_lim = theta[0] * (
            np.dot(
                z_alpha,
                np.divide(
                    np.sqrt(2 * theta[1] * h_0 ** 2),
                    theta[0]
                )
            ) + 1 + np.divide(
                theta[1] * h_0 * (h_0 - 1),
                theta[0] ** 2)
        ) ** (1 / h_0)

    limits['Q'] = Q_lim

    return limits

def fault_detection(data, start_cal, end_cal, min_var_exp, alpha):
    # Scaler instance to normalize the data
    df = data.copy(deep=True)
    scaler = StandardScaler()
    # building the pca model by fitting to a segment of good quality data
    start_cal = pd.to_datetime(start_cal)
    end_cal = pd.to_datetime(end_cal)
    raw_cal_dat = df.loc[start_cal:end_cal].to_numpy()
    norm_cal_dat = scaler.fit_transform(raw_cal_dat)

    model, svd_feat = build_pca(norm_cal_dat, min_var_exp)

    # Setting the limit value for each test
    limits = calc_limits(norm_cal_dat, model['n_comp'], alpha, svd_feat['eigenvalues'])

    # extract the std of the principal components
    limits['pc_std'] = model['t_hat_stdev']

    # save the start and end of calibration to limits dictionnary
    limits['start_cal'] = start_cal
    limits['end_cal'] = end_cal

    # Applying the model transformations to the complete data set
    raw_tot_dat = df.to_numpy()
    norm_tot_dat = scaler.transform(raw_tot_dat)
    pc_tot_dat = np.dot(norm_tot_dat, model['loading_mat'])
    for i in range(model['n_comp']):
        df['pc_' + str(i + 1)] = pc_tot_dat[:, i]

    # calculate the T**2 and Q statistics of the entire data set
    df['T2'], df['Q'] = calc_stats(norm_tot_dat, model['loading_mat'], model['lambda_hat'])

    # Count the number of faults for each
    df['fault_count'] = 0
    df.loc[df['T2'] > limits['T2'], ['fault_count']] += 1
    df.loc[df['Q'] > limits['Q'], ['fault_count']] += 1

    contrib = np.zeros(model['loading_mat'].T.shape)
    abs_loading_t = np.abs(model['loading_mat'].T)
    for i in range(abs_loading_t.shape[0]):
        sum_line = abs_loading_t[i].sum()
        for j in range(abs_loading_t.shape[1]):
            contrib[i, j] = abs_loading_t[i, j] / sum_line
    contrib = contrib.T
    return df, limits, contrib

'''# Debugging code
path = '../sample_data/influent3.csv'
raw_data = pd.read_csv(path, sep=';')
raw_data.datetime = pd.to_datetime(raw_data.datetime)
raw_data.set_index('datetime', inplace=True, drop=True)
renaming = {}
for name in raw_data.columns:
    renaming[name] = 'raw-{}'.format(name)
raw_data.rename(index=str, columns=renaming)
df = raw_data.copy(deep=True)

start_cal = '15 January 2018'
end_cal = '15 February 2018'
min_var_exp = 0.8
alpha = 0.95
data, limits, loading_mat = fault_detection(df, start_cal, end_cal, min_var_exp, alpha)
# PlottingTools.show_pca_mpl(data, limits)
PlottingTools.show_multi_output_mpl(df)
'''

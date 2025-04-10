import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Ellipse
from pathlib import Path
from scipy.stats.mstats import gmean
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Union


normal_color = 'dodgerblue'
tumor_color = 'orangered'
old_color = 'blue'
alz_color = 'red'
fontsize = 14
tick_fontsize = 12

gbm_sample_path: Path = Path(r'data/TCGA-GBM')
gbm_path: Path = gbm_sample_path / 'data'
alz_path: Path = Path(r'data/ALZ')

outputpath: Path = Path(r'Figures and tables')
if not outputpath.is_dir():
    outputpath.mkdir()


def read_alz_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Loading DonorInformation
    DonorInformation = pd.read_csv(alz_path / 'DonorInformation.csv', sep=';')

    # Masks to get no dementia and Alzheimer disease samples
    no_dementia_mask = DonorInformation.dsm_iv_clinical_diagnosis == 'No Dementia'
    alz_mask = DonorInformation.dsm_iv_clinical_diagnosis == "Alzheimer's Disease Type"

    # Getting donor_id
    NoDementia_donor_id = DonorInformation.loc[no_dementia_mask, 'donor_id']
    Alz_donor_id = DonorInformation.loc[alz_mask, 'donor_id']

    # Loading ColumnSample
    ColumnsSample = pd.read_csv(alz_path / 'columns-samples.csv', sep=';', index_col=1)

    NoDementia_CS = ColumnsSample.loc[NoDementia_donor_id]
    Alz_CS = ColumnsSample.loc[Alz_donor_id]

    # Masks to get only Forewhite matter (FWM)
    nd_fwm_mask = NoDementia_CS.structure_acronym == 'FWM'
    alz_fwm_mask = Alz_CS.structure_acronym == 'FWM'

    # Getting rnaseq_profile_id
    NoDementia_FWM_id = NoDementia_CS.loc[nd_fwm_mask, 'rnaseq_profile_id'].astype(str)
    Alz_FWM_id = Alz_CS.loc[alz_fwm_mask, 'rnaseq_profile_id'].astype(str)

    # Loading FPKM file
    fpkm = pd.read_csv(alz_path / 'fpkm_table_normalized.csv', index_col=0)

    # Getting no dementia and alz data
    old = fpkm[NoDementia_FWM_id].to_numpy().T + 0.1
    alz = fpkm[Alz_FWM_id].to_numpy().T + 0.1

    nd_fwm_mask = DonorInformation.donor_id.isin(NoDementia_FWM_id.index)
    alz_fwm_mask = DonorInformation.donor_id.isin(Alz_FWM_id.index)

    old_age = DonorInformation.age[nd_fwm_mask].to_numpy()
    alz_age = DonorInformation.age[alz_fwm_mask].to_numpy()

    return old, alz, old_age, alz_age


def fig_2_2(old, alz):
    marker_size = 15
    alpha_value = 0.7

    k = old.shape[0]
    old += 0.1
    alz += 0.1
    ref = gmean(old)
    data = np.log2(np.concatenate([old, alz])/ref)

    # pca = PCA(n_components=2)
    # pcs = pca.fit_transform(data)
    # var = pca.explained_variance_ratio_
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    eigenvalues: np.ndarray = s**2/data.shape[0]
    var: np.ndarray = eigenvalues / eigenvalues.sum()
    eigenvectors: np.ndarray= Vt
    projection: np.ndarray = np.dot(Vt, data.T)

    old_pcs = projection[:2,:k].T
    alz_pcs = projection[:2,k:].T

    # old_pcs = pcs[:k]
    # alz_pcs = pcs[k:]

    old_mean_x, old_mean_y = old_pcs.mean(axis=0)
    old_std_x, old_std_y = old_pcs.std(axis=0)
    alz_mean_x, alz_mean_y = alz_pcs.mean(axis=0)
    alz_std_x, alz_std_y = alz_pcs.std(axis=0)

    l = 3
    old_ell = Ellipse((old_mean_x, -old_mean_y),
                      width=l*old_std_x, height=l*old_std_y,
                      edgecolor=old_color, facecolor='none',
                      linewidth=1.5, linestyle='--')
    alz_ell = Ellipse((alz_mean_x, -alz_mean_y),
                      width=l*alz_std_x, height=l*alz_std_y,
                      edgecolor=alz_color, facecolor='none',
                      linewidth=1.5, linestyle='--')

    gr = (1+np.sqrt(5))/2
    h=3.5
    w=h*gr
    fig, ax = plt.subplots(figsize=(w, h))
    ax.scatter(old_pcs[:, 0], -old_pcs[:, 1], label='O',
               c=old_color, s=marker_size, alpha=alpha_value, marker='s')
    ax.scatter(alz_pcs[:, 0], -alz_pcs[:, 1], label='EA',
               c=alz_color, s=marker_size, alpha=alpha_value)
    ax.add_patch(old_ell)
    ax.add_patch(alz_ell)

    ax.set_xlabel(f'PC1 ({var[0]:.2%})')
    ax.set_ylabel(f'PC2 ({var[1]:.2%})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputpath / 'pca_o_to_ad_1.pdf', bbox_inches='tight')

    return old_pcs, alz_pcs


def _set_coord(pcs, age):
    x = []
    y = []
    for i in set(age):
        x.append(int(i if i.isdecimal() else i.replace('+', '-').split('-')[0]))
        mask = age == i
        y.append(pcs[mask, 0].mean())

    return x, y


def _set_old(pcs, age):
    p1x, p2x, p3x, p4x = [], [], [], []
    p1y, p2y, p3y, p4y = [], [], [], []

    for i, j in zip(age, pcs):
        x = int(i if i.isdecimal() else i.replace('+', '-').split('-')[0])
        if x < 84:
            # p1x.append(x)
            p1x.append(82)
            p1y.append(j[0])
        elif x < 90:
            # p2x.append(x)
            p2x.append(87)
            p2y.append(j[0])
        elif x < 95:
            # p3x.append(x)
            p3x.append(92)
            p3y.append(j[0])
        else:
            # p4x.append(x)
            p4x.append(96)
            p4y.append(j[0])

    # x = np.array([np.mean(p1x), np.mean(p2x), np.mean(p3x), np.mean(p4x)])
    # y = np.array([np.mean(p1y), np.mean(p2y), np.mean(p3y), np.mean(p4y)])

    return p1x, p2x, p3x, p4x, p1y, p2y, p3y, p4y


def _set_alz(pcs, age):
    p1x, p2x = [], []
    p1y, p2y = [], []

    for i, j in zip(age, pcs):
        x = int(i if i.isdecimal() else i.replace('+', '-').split('-')[0])
        if x < 90:
            p1x.append(84)
            p1y.append(j[0])
        else:
            p2x.append(96)
            p2y.append(j[0])

    x = np.array([np.mean(p1x), np.mean(p2x)])
    y = np.array([np.mean(p1y), np.mean(p2y)])

    print(np.mean(y))

    return p1x, p2x, p1y, p2y


def fig_2_3(old_pcs, alz_pcs, old_age, alz_age):
    o1x, o2x, o3x, o4x, o1y, o2y, o3y, o4y = _set_old(old_pcs, old_age)
    a1x, a2x, a1y, a2y = _set_alz(alz_pcs, alz_age)

    x_old = np.array([np.mean(o1x), np.mean(o2x), np.mean(o3x), np.mean(o4x)])
    y_old = np.array([np.mean(o1y), np.mean(o2y), np.mean(o3y), np.mean(o4y)])
    x_alz = np.array([np.mean(a1x), np.mean(a2x)])
    y_alz = np.array([np.mean(a1y), np.mean(a2y)])


    model = LinearRegression()
    model.fit(x_old.reshape((-1, 1)), y_old)
    y_fit = model.predict(x_old.reshape((-1, 1)))
    slope, = model.coef_
    intercept = model.intercept_
    r2 = model.score(x_old.reshape((-1, 1)), y_old)

    x1 = np.linspace(70, 110, 100)
    y1 = np.array([intercept+i*slope for i in x1])
    print(f'old={slope}*age {intercept}')

    marker_size = 15
    alpha_value = 0.7
    gr = (1+np.sqrt(5))/2
    h=3.5
    w=h*gr
    fig, ax = plt.subplots(figsize=(w, h))

    ax.scatter(x_old, y_old, label='O',
               c=old_color, s=marker_size, alpha=alpha_value, marker='s')
    ax.scatter(x_alz, y_alz, label='EA',
               c=alz_color, s=marker_size, alpha=alpha_value)
    ax.axhline(y=27.38, color=alz_color, linestyle='--')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.plot(x1, y1, '--', c=old_color)

    ax.set_xlabel('Edad')
    ax.set_ylabel(r'$\left\langle x_1 \right\rangle$')
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlim((78, 101))
    ax.set_ylim((-40, 30))
    fig.tight_layout()
    fig.savefig(outputpath / 'O_to_AD_1.pdf', bbox_inches='tight')

    return (o1y, o2y, o3y, o4y), (a1y, a2y)


def fig_2_4(oldy, alzy):
    alz = np.concatenate(alzy)
    old1, old2, old3, old4 = oldy

    bw = 0.42
    gr = (1+np.sqrt(5))/2
    h=4.8
    w=h*gr
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(w, h))

    ax1 = axs[0, 0]
    sns.kdeplot(alz, color=alz_color, fill=True, ax=ax1, label='EA', bw_adjust=bw)
    sns.kdeplot(old1, color=old_color, fill=True, ax=ax1, label='O, 79 < edad < 84', bw_adjust=bw)
    ax1.legend()
    ax1.set_ylabel('Densidad de probabilidad')
    ax1.set_xlim((-200, 150))
    ax1.set_ylim((0.0, 0.014))

    ax2 = axs[0, 1]
    sns.kdeplot(alz, color=alz_color, fill=True, ax=ax2, label='EA', bw_adjust=bw)
    sns.kdeplot(old2, color=old_color, fill=True, ax=ax2, label='O, 84 < edad < 90', bw_adjust=bw)
    ax2.legend()

    ax3 = axs[1, 0]
    sns.kdeplot(alz, color=alz_color, fill=True, ax=ax3, label='EA', bw_adjust=bw)
    sns.kdeplot(old3, color=old_color, fill=True, ax=ax3, label='O, 90 < edad < 95', bw_adjust=bw)
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('Densidad de probabilidad')
    ax3.legend()

    ax4 = axs[1, 1]
    sns.kdeplot(alz, color=alz_color, fill=True, ax=ax4, label='EA', bw_adjust=bw)
    sns.kdeplot(old4, color=old_color, fill=True, ax=ax4, label='O, 95 < edad < 101', bw_adjust=bw)
    ax4.set_xlabel('PC1')
    ax4.legend()

    fig.savefig(outputpath / 'suppl_otoad.pdf', bbox_inches='tight')


def main():
    old, alz, old_age, alz_age = read_alz_data()
    old_pcs, alz_pcs = fig_2_2(old, alz)
    oldy, alzy = fig_2_3(old_pcs, alz_pcs, old_age, alz_age)
    fig_2_4(oldy, alzy)
    plt.show()

if __name__ == '__main__':
    main()
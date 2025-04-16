import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patheffects import withStroke
from pathlib import Path
from scipy.stats.mstats import gmean
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


def read_gbm_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The "sample.xls" file has the information about each sample
    sample: pd.DataFrame = pd.read_excel(gbm_sample_path / 'sample.xls')

    # Masks to get the normal and tumor samples
    normal_mask: pd.Series = sample['Sample Type'] == 'Solid Tissue Normal'
    tumor_mask: pd.Series = sample['Sample Type'] != 'Solid Tissue Normal'

    # Reading first file to get the ensembl ids list
    column_names: List = ['gene_id', 'value']
    file: pd.DataFrame = pd.read_table(gbm_path / sample['File Name'][0], names=column_names)

    # Identifier of each gene in ensembl format
    genes_id: np.ndarray = file['gene_id'].to_numpy()
    genes_id = np.vectorize(lambda x: x.split('.')[0])(genes_id)

    # Loading data
    def get_data(row: pd.Series) -> pd.Series:
        filename = row['File Name']
        file = pd.read_table(gbm_path / filename, names=['gene_id', 'value'])
        return file['value']

    data: np.ndarray = sample.apply(get_data, axis=1).to_numpy()
    data += 0.1

    return data[normal_mask], data[tumor_mask], genes_id


def read_alz_data() -> Tuple[np.ndarray, np.ndarray]:
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

    return old, alz


def get_filters(genes_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Loading the modified rows_genes file with the ensembl ids
    rows_genes = pd.read_excel(alz_path / 'rows_genes2.xlsx')

    # Get valids ensembl ids
    ensembls_id = rows_genes.ensembl.dropna()

    # Auxuliar dataframe for to get the the normal and tumor indices
    df1 = pd.DataFrame(data={'i': range(genes_id.size)}, index=genes_id)

    # Get the indices
    index_oa = ensembls_id.index.to_numpy()
    index_nt = df1.loc[ensembls_id, 'i'].to_numpy()

    # Getting corresponding symbols
    symbols = rows_genes.loc[index_oa, 'gene_symbol']

    return index_nt, index_oa, symbols.to_numpy()


def save_n_comp(fname: str, vect: np.ndarray, genes: np.ndarray, n: Optional[int] = None) -> None:
    n = n or len(vect)  

    args: np.ndarray = np.argsort(-np.abs(vect))
    file = open(fname, 'w')

    for arg in args[:n]:
        file.write(f'{genes[arg]}\t{vect[arg]}\n')
    file.close()


def figure_1b(normal: np.ndarray, tumor: np.ndarray, old: np.ndarray, alz: np.ndarray):
    normal_center = normal.mean(axis=1)
    tumor_center = tumor.mean(axis=1)
    old_center = old.mean(axis=1)
    alz_center = alz.mean(axis=1)

    fig1b, ax1b = plt.subplots()

    ax1b.scatter(*normal_center, marker='o', c=normal_color, s=50, label='N')
    ax1b.scatter(*tumor_center, marker='o', c=tumor_color, s=50, label='GB')
    ax1b.scatter(*old_center, marker='o', c=old_color, s=50, label='O')
    ax1b.scatter(*alz_center, marker='o', c=alz_color, s=50, label='EA')

    # Arrow from normal to tumor center
    ax1b.annotate('', xy=tumor_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=-1, armA=0,"
                                  "angleB=100, armB=60, rad=50"))

    # Arrow from normal to old center
    ax1b.annotate('', xy=old_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=1, armA=0,"
                                  "angleB=-100, armB=90, rad=80"))

    # Arrow from normal to alzheimer center
    ax1b.annotate('', xy=alz_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=2, armA=0,"
                                  "angleB=-115, armB=90, rad=90"))
    
    # Arrow from old to alzheimer center
    fs = 12
    fs2 = 11
    ax1b.annotate('', xy=alz_center, xytext=old_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9))
    
    ax1b.annotate('N', xy=normal_center + (-4, -10),
                  c='k',  size=fs, va='top', ha='right', weight='bold')

    ax1b.annotate('GB', xy=tumor_center + (-10, 25),
                  c='k', size=fs, va='top', ha='right', weight='bold')

    ax1b.annotate('O', xy=old_center + (10, -12),
                  c='k', size=fs, va='center', ha='left',  weight='bold')

    ax1b.annotate('EA', xy=alz_center + (-15, 0),
                  c='k', size=fs, va='top', ha='right', weight='bold')

    ax1b.annotate('EA\nanticipada',  xy=alz_center/2 + (30, -40), weight='bold',
                  ha='center', rotation=20, size=fs2)

    ax1b.annotate('Envejecimiento', xy=old_center/2 + (130, -90), weight='bold',
                  ha='right', va='bottom', rotation=35, size=fs2)

    ax1b.text(*((old_center + alz_center)/2 + (-0, 15)), 
              'EA\ntardía', va='bottom', ha='center', weight='bold', size=fs2)

    ax1b.set_xlabel('PC1', fontsize=fontsize)
    ax1b.set_ylabel('PC2', fontsize=fontsize)
    # ax1b.set_title('b', fontsize=fontsize)
    ax1b.tick_params(axis='x', labelsize=tick_fontsize)
    ax1b.tick_params(axis='y', labelsize=tick_fontsize)
    ax1b.set(xlim=(-20, 270), ylim=(-180, 290))
    plt.tight_layout()

    fig1b.savefig(outputpath / 'Fig_1b.pdf',  bbox_inches='tight')


def graphical_abstract(normal: np.ndarray, tumor: np.ndarray, old: np.ndarray, alz: np.ndarray):
    normal_center = normal.mean(axis=1)
    tumor_center = tumor.mean(axis=1)
    old_center = old.mean(axis=1)
    alz_center = alz.mean(axis=1)

    fig_ga, ax_ga = plt.subplots()

    ax_ga.scatter(*normal_center, marker='o', c=normal_color, s=50, label='N')
    ax_ga.scatter(*tumor_center, marker='o', c=tumor_color, s=50, label='GB')
    ax_ga.scatter(*old_center, marker='o', c=old_color, s=50, label='O')
    ax_ga.scatter(*alz_center, marker='o', c=alz_color, s=50, label='AD')

    # Arrow from normal to tumor center
    ax_ga.annotate('', xy=tumor_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=-1, armA=0,"
                                  "angleB=100, armB=60, rad=50"))

    # Arrow from normal to old center
    ax_ga.annotate('', xy=old_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=1, armA=0,"
                                  "angleB=-100, armB=90, rad=80"))

    # Arrow from normal to alzheimer center
    ax_ga.annotate('', xy=alz_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=2, armA=0,"
                                  "angleB=-115, armB=90, rad=90"))
    
    # Arrow from old to alzheimer center
    ax_ga.annotate('', xy=alz_center, xytext=old_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9))
    
    ax_ga.annotate('N', xy=normal_center + (-4, -10),
                  c='k',  size=12, va='top', ha='right', weight='bold')

    ax_ga.annotate('GB', xy=tumor_center + (-10, 25),
                  c='k', size=12, va='top', ha='right', weight='bold')

    ax_ga.annotate('O', xy=old_center + (10, -12),
                  c='k',  size=12, va='center', ha='left',  weight='bold')

    ax_ga.annotate('AD', xy=alz_center + (-15, 0),
                  c='k', size=12, va='top', ha='right', weight='bold')

    ax_ga.annotate('Early\nAD',  xy=alz_center/2 + (50, 15), weight='bold')

    # ax_ga.text(285, 75, "Aging", ha='left', va='top', weight='bold')
    ax_ga.annotate('Aging', xy=old_center/2 + (130, -5), weight='bold', ha='right', va='bottom')

    ax_ga.text(*((old_center + alz_center)/2 + (-5, 15)), 
              'Late AD', va='bottom', ha='center', weight='bold')

    # ax_ga.tick_params(axis='x', labelsize=tick_fontsize)
    ax_ga.tick_params(axis='both', which='both', length=0)
    ax_ga.set_xticklabels([])
    ax_ga.set_yticklabels([])
    ax_ga.set_xlabel('Aging axis', fontsize=fontsize)
    ax_ga.set_ylabel('GB vs. AD axis', fontsize=fontsize)
    ax_ga.set(xlim=(-20, 270), ylim=(-180, 290))

    fig_ga.savefig(outputpath / 'graphical_abstract.pdf',  bbox_inches='tight')


def normal_dist(x: Union[float, np.ndarray],
                        x_mean: Union[float, np.ndarray],
                        sigma: Union[float, np.ndarray]
                        ) -> Union[float, np.ndarray]:
    return np.exp(-np.power(x - x_mean, 2) / (2 * sigma*2))


def figure_1c(normal: np.ndarray, tumor: np.ndarray, alz: np.ndarray):
    x = np.linspace(-180, 400, 100)
    y = np.linspace(-250, 330, 100)
    X, Y = np.meshgrid(x, y)
    normal_center, normal_std = normal.mean(axis=1), 150 * normal.std(axis=1)
    tumor_center, tumor_std = tumor.mean(axis=1), 70 * tumor.std(axis=1)
    alz_center, alz_std = alz.mean(axis=1), 100 * alz.std(axis=1)

    Z_normal = normal_dist(X, normal_center[0], normal_std[0]) * normal_dist(Y, normal_center[1], normal_std[1])
    Z_tumor = normal_dist(X, tumor_center[0], tumor_std[0]) * normal_dist(Y, tumor_center[1], tumor_std[1])
    Z_alz = normal_dist(X, alz_center[0], alz_std[0]) * normal_dist(Y, alz_center[1], alz_std[1])

    Z = 15 * Z_normal + 30 * Z_tumor + 3 * Z_alz

    fig1c, ax1c = plt.subplots()

    white_outline = withStroke(linewidth=3, foreground='white')

    # # Calculate the minimum and maximum of Z
    # z_min, z_max = 0, Z.max()
    # # Define a threshold below which you want more levels.
    # # Here, we choose a value at 20% of the range above the minimum.
    # low_threshold = z_min + 0.01 * (z_max - z_min)
    
    # # Create many levels for the low value region and fewer for the high region.
    # low_levels = np.linspace(z_min, low_threshold, 5)  # Denser in low values
    # high_levels = np.linspace(low_levels[-1], z_max, 75)    # Sparser in the remaining range
    # low_levels = low_levels[:-1]

    # print(f'{low_levels = }')
    # print(f'{high_levels = }')
    
    # # Combine the two sets of levels into one list.
    # levels = np.concatenate([low_levels, high_levels])
    # print(f'{np.min(Z) = }, {np.max(Z) = }')

    # # Avoid taking the logarithm of zero:
    # nonzero_Z = Z[Z > 0]
    # if nonzero_Z.size == 0:
    #     # If Z somehow has no positive values, provide a fallback.
    #     min_val = 1e-10
    # else:
    #     min_val = nonzero_Z.min()
    # max_val = Z.max()

    # # Create a logarithmically spaced progression of levels.
    # # The number of levels (here, 200) can be adjusted for your preferred level density.
    # levels = np.logspace(np.log10(min_val), np.log10(max_val), 150)

    # --- Build custom levels ---
    # Ensure that only positive values are used for the log spacing.
    nonzero_Z = Z[Z > 0]
    if nonzero_Z.size == 0:
        raise ValueError("Z does not contain any positive values; cannot use log spacing.")
    min_val = nonzero_Z.min()
    max_val = Z.max()

    # Choose a threshold to switch from log to linear spacing.
    log_threshold = min_val + 0.0135 * (max_val - min_val)

    # Create the log-spaced levels in the "low" region.
    num_log_levels = 4  # Number of levels in the log-spaced (low) region
    log_levels = np.logspace(np.log10(min_val), np.log10(log_threshold), num=num_log_levels)

    # Calculate the step between the last two log levels.
    gap = log_levels[-1] - log_levels[-2]

    # Create linear-spaced levels in the high region.
    # The first level of the high region will be (last-log-level + gap) to maintain a continuous spacing.
    linear_levels = np.arange(log_levels[-1] + gap, max_val, gap)
    
    # Make sure the very last value is included.
    if linear_levels.size == 0 or linear_levels[-1] < max_val:
        linear_levels = np.append(linear_levels, max_val)

    # Combine the log and linear parts.
    levels = np.concatenate([log_levels, linear_levels])
    # --- End build custom levels ---
    
    ax1c.contour(X, Y, Z, levels=levels, cmap='gray')

    ax1c.annotate('N', xy=normal_center, c=normal_color, va='bottom', ha='left', size=12,
                  weight='bold', path_effects=[white_outline])
    ax1c.annotate('GB', xy=tumor_center, c=tumor_color, va='bottom', ha='left', size=12,
                  weight='bold', path_effects=[white_outline])
    ax1c.annotate('EA', xy=alz_center, c=alz_color, va='bottom', ha='left', size=12,
                  weight='bold', path_effects=[white_outline])
    
    # ax1c.set_title('c', fontsize=fontsize)
    ax1c.set_xlabel('PC1', fontsize=fontsize)
    ax1c.set_ylabel('PC2', fontsize=fontsize)
    ax1c.tick_params(axis='x', labelsize=tick_fontsize)
    ax1c.tick_params(axis='y', labelsize=tick_fontsize)
    ax1c.set_xlim((-180, 400))
    ax1c.set_ylim((-250, 330))

    fig1c.savefig(outputpath / 'Fig_1c.pdf', bbox_inches='tight')


def pca_analysis(normal: np.ndarray, tumor: np.ndarray,
                 old: np.ndarray, alz: np.ndarray, genes: np.ndarray) -> None:
    data: np.ndarray = np.concatenate((normal, tumor, old, alz))
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    eigenvalues: np.ndarray = s**2/data.shape[0]
    eigenvalues_normalized: np.ndarray = eigenvalues / eigenvalues.sum()
    eigenvectors: np.ndarray= Vt
    projection: np.ndarray = np.dot(Vt, data.T)

    i1 = normal.shape[0]
    i2 = i1 + tumor.shape[0]
    i3 = i2 + old.shape[0]
    
    # Figures
    # Fig 1a (PC1-PC2)
    fig1a, ax1a = plt.subplots()

    marker_size = 15
    alpha_value = 0.7

    ax1a.scatter(projection[0, :i1], projection[1, :i1], label="N",
                 c=normal_color, s=marker_size, alpha=alpha_value)
    
    ax1a.scatter(projection[0, i1:i2], projection[1, i1:i2], label="GB",
                 c=tumor_color, s=marker_size, marker='s', alpha=alpha_value)
    
    ax1a.scatter(projection[0, i2:i3], projection[1, i2:i3], label="O",
                 c=old_color, s=marker_size, marker='s', alpha=alpha_value)
    
    ax1a.scatter(projection[0, i3:], projection[1, i3:], label="EA",
                 c=alz_color, s=marker_size, alpha=alpha_value)

    # ax1a.set_title('a')
    ax1a.set_xlabel(f'PC1 ({eigenvalues_normalized[0]*100:.2f}%)', fontsize=fontsize)
    ax1a.set_ylabel(f'PC2 ({eigenvalues_normalized[1]*100:.2f}%)', fontsize=fontsize)
    ax1a.legend(fontsize=fontsize)
    ax1a.tick_params(axis='x', labelsize=tick_fontsize)
    ax1a.tick_params(axis='y', labelsize=tick_fontsize)
    plt.tight_layout()

    # Fig 1b
    figure_1b(projection[:2, :i1], projection[:2, i1:i2],
              projection[:2, i2:i3], projection[:2, i3:])
    
    # Fig 1c
    figure_1c(projection[:2, :i1], projection[:2, i1:i2], projection[:2, i3:])

    # Graphical Abstract
    # graphical_abstract(projection[:2, :i1], projection[:2, i1:i2],
    #                    projection[:2, i2:i3], projection[:2, i3:])

    # Exporting data
    fig1a.savefig(outputpath / 'Fig_1a.pdf', bbox_inches='tight')

    save_n_comp(outputpath / 'PC1.txt', eigenvectors[0], genes, 100)
    save_n_comp(outputpath / 'PC2.txt', eigenvectors[1], genes, 100)


def figure_1d(normal: np.ndarray, tumor: np.ndarray, alz: np.ndarray, genes: List, symbols: np.ndarray):
    fig1d, ax1d = plt.subplots()
    
    gene_id = []
    gene_name = []
    for g in genes:
        gene_id.append(np.argwhere(symbols == g)[0, 0])
        gene_name.append(g)
    
    n_genes = len(gene_id)

    groups = (['GB'] * tumor.shape[0] * n_genes
              + ['N'] * normal.shape[0] * n_genes
              + ['EA'] * alz.shape[0] * n_genes)
    
    columns_tumor = np.concatenate([tumor[:, gid] for gid in gene_id])
    columns_normal = np.concatenate([normal[:, gid] for gid in gene_id])
    columns_alz = np.concatenate([alz[:, gid] for gid in gene_id])
    columns = np.concatenate([
        columns_tumor,
        columns_normal,
        columns_alz,
    ])
    
    categories = []
    for gn in gene_name:
        categories += [gn] * tumor.shape[0]
    for gn in gene_name:
        categories += [gn] * normal.shape[0]
    for gn in gene_name:
        categories += [gn] * alz.shape[0]

    sns.violinplot(x=groups, y=columns, hue=categories, inner='quart', ax=ax1d)
    ax1d.axhline(y=0, color='gray', linestyle='-.', linewidth=1.1, alpha=0.7)
    ax1d.set_ylabel('$log_2(e/e_{ref})$', fontsize=fontsize)
    ax1d.tick_params(axis='x', labelsize=fontsize)
    ax1d.tick_params(axis='y', labelsize=tick_fontsize)
    # ax1d.set_title('d', fontsize=fontsize)
    plt.tight_layout()

    fig1d.savefig(outputpath / 'Fig_1d.pdf', bbox_inches='tight')


def figure_1d_2(normal: np.ndarray, tumor: np.ndarray, alz: np.ndarray, genes: List, symbols: np.ndarray):
    gr = (1+np.sqrt(5))/2
    h=4.8
    w=h*gr
    fig1d, ax1d = plt.subplots(figsize=(w, h))
    
    gene_id = []
    gene_name = []
    for g in genes:
        gene_id.append(np.argwhere(symbols == g)[0, 0])
        gene_name.append(g)
    
    n_genes = len(gene_id)

    groups = (['GB'] * tumor.shape[0] * n_genes
              + ['N'] * normal.shape[0] * n_genes
              + ['EA'] * alz.shape[0] * n_genes)
    
    columns_tumor = np.concatenate([tumor[:, gid] for gid in gene_id])
    columns_normal = np.concatenate([normal[:, gid] for gid in gene_id])
    columns_alz = np.concatenate([alz[:, gid] for gid in gene_id])
    columns = np.concatenate([
        columns_tumor,
        columns_normal,
        columns_alz,
    ])
    
    categories = []
    for gn in gene_name:
        categories += [gn] * tumor.shape[0]
    for gn in gene_name:
        categories += [gn] * normal.shape[0]
    for gn in gene_name:
        categories += [gn] * alz.shape[0]

    sns.violinplot(x=groups, y=columns, hue=categories, inner='quart', ax=ax1d)
    ax1d.axhline(y=0, color='gray', linestyle='-.', linewidth=1.1, alpha=0.7)
    ax1d.set_ylabel('$log_2(e/e_{ref})$', fontsize=fontsize)
    ax1d.tick_params(axis='x', labelsize=fontsize)
    ax1d.tick_params(axis='y', labelsize=tick_fontsize)
    # ax1d.set_title('d', fontsize=fontsize)
    plt.tight_layout()

    fig1d.savefig(outputpath / 'suppl2.pdf', bbox_inches='tight')


def main():
    # Loading data
    normal, tumor, genes_id = read_gbm_data()
    old, alz = read_alz_data()

    # Getting filters and gene symbols
    index_nt, index_oa, symbols = get_filters(genes_id)

    normal = normal[:, index_nt]
    tumor = tumor[:, index_nt]
    old = old[:, index_oa]
    alz = alz[:, index_oa]

    # Normalizing data
    normal = 1_000_000 * (normal / normal.sum(axis=1)[:, np.newaxis])
    tumor = 1_000_000 * (tumor / tumor.sum(axis=1)[:, np.newaxis])
    old = 1_000_000 * (old / old.sum(axis=1)[:, np.newaxis])
    alz = 1_000_000 * (alz / alz.sum(axis=1)[:, np.newaxis])

    # Fold change
    ref = gmean(normal)
    normal = np.log2(normal / ref)
    tumor = np.log2(tumor / ref)
    old = np.log2(old / ref)
    alz = np.log2(alz / ref)

    # geometry_analysis(normal, tumor, old, alz, symbols)
    pca_analysis(normal, tumor, old, alz, symbols)
    figure_1d(normal, tumor, alz, ['MMP9', 'BCYRN1'], symbols)
    figure_1d_2(normal, tumor, alz, ['MAPT', 'APP', 'APOE', 'IDH1', 'IDH2', 'MKI67', 'ATRX'], symbols)
    plt.show()


if __name__ == "__main__":
    main()

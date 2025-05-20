import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

outputpath: Path = Path(r'Figures and tables')
if not outputpath.is_dir():
    outputpath.mkdir()


def read_pcs():
    pc1 = pd.read_table(outputpath / 'PC1.txt', names=['gene', 'value'])
    pc2 = pd.read_table(outputpath / 'PC2.txt', names=['gene', 'value'])

    return pc1, pc2


def pci_graph(pc, pcn):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Enumerate through values and genes
    colors = ['r' if v > 0 else 'b' for v in pc.value]

    # Plot points and lines
    for i, (v, color) in enumerate(zip(pc.value, colors), start=1):
        ax.scatter(i, v, color=color)
        ax.plot([i, i], [0, v], color=color)

    # Set x-ticks and labels
    ax.set_xticks(range(1, len(pc.gene) + 1))
    ax.set_xticklabels(pc.gene, rotation=70, fontsize=8)

    for tick_label, color in zip(ax.get_xticklabels(), colors):
        tick_label.set_color(color)

    # Labels and grid
    ax.set_xlabel('Gen')
    ax.set_ylabel(f'Peso en {pcn}')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    # ax.grid(True, linestyle=':', linewidth=0.5, axis='y')

    plt.tight_layout()
    fig.savefig(outputpath / f'{pcn}.pdf')
        


def main():
    pc1, pc2 = read_pcs()
    pci_graph(pc1.iloc[:30], 'PC1')
    pci_graph(pc2.iloc[:30], 'PC2')
    plt.show()


if __name__ == '__main__':
    main()
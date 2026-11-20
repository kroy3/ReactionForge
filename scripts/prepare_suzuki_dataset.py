#!/usr/bin/env python3
"""
Prepare the Perera et al. Suzuki-Miyaura dataset for ReactionForge
===================================================================
Converts the 5,760-reaction high-throughput Suzuki-Miyaura dataset from

    D. Perera et al., "A platform for automated nanomole-scale reaction
    screening and micromole-scale synthesis in flow", Science 359, 429 (2018)

into ReactionForge's CSV format (reactant_smiles, boronic_acid_smiles,
ligand_smiles, base_smiles, solvent_smiles, product_smiles, yield).

The data is not redistributed with ReactionForge. By default the script
downloads the paper's supplementary spreadsheet (Data File S1) and the
standard random train/test splits from the rxn_yields repository
(Schwaller et al., Mach. Learn.: Sci. Technol. 2, 015016, 2021), which most
yield-prediction papers report results on.

Usage:
    # Whole dataset as one CSV
    python scripts/prepare_suzuki_dataset.py --output data/suzuki_perera.csv

    # Standard random split k (0-9): first 70% train, last 30% test
    python scripts/prepare_suzuki_dataset.py --split 0 --output_dir data/suzuki_split0

Component structures follow the reaction SMILES used in rxn_yields, so
results are comparable with work that uses that representation. dppf and
dtbpf are represented there by a single phosphine-substituted Cp ring plus
Fe, not the full ferrocene; this script keeps that convention.
"""

import argparse
import io
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.dataset import suzuki_product

RXN_YIELDS = 'https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Suzuki-Miyaura'
XLSX_URL = f'{RXN_YIELDS}/aap9112_Data_File_S1.xlsx'
SPLIT_URL = RXN_YIELDS + '/random_splits/random_split_{k}.tsv'
TRAIN_FRACTION = 0.7  # rxn_yields convention: first 70% of each split file trains

# Coupling partner 1 ("Reactant_1_Short_Hand"): quinoline electrophiles 1a-1d,
# quinoline boron reagents 1e-1g (which couple with the indazole bromide 2d)
REACTANT_1 = {
    '1a, 6-Cl-Q': 'Clc1ccc2ncccc2c1',
    '1b, 6-Br-Q': 'Brc1ccc2ncccc2c1',
    '1c, 6-OTf-Q': 'O=S(=O)(Oc1ccc2ncccc2c1)C(F)(F)F',
    '1d, 6-I-Q': 'Ic1ccc2ncccc2c1',
    '1e, 6-BOH2-Q': 'OB(O)c1ccc2ncccc2c1.Cl',
    '1f, 6-BPin-Q': 'CC1(C)OB(c2ccc3ncccc3c2)OC1(C)C',
    '1g, 6-BF3K-Q': 'F[B-](F)(F)c1ccc2ncccc2c1.[K+]',
}
# Coupling partner 2 ("Reactant_2_Name"): indazole boron reagents 2a-2c, bromide 2d
REACTANT_2 = {
    '2a, Boronic Acid': 'Cc1ccc2c(cnn2C2CCCCO2)c1B(O)O',
    '2b, Boronic Ester': 'Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1',
    '2c, Trifluoroborate': 'Cc1ccc2c(cnn2C2CCCCO2)c1[B-](F)(F)F.[K+]',
    '2d, Bromide': 'Cc1ccc2c(cnn2C2CCCCO2)c1Br',
}
LIGANDS = {
    'AmPhos': 'CN(C)c1ccc(P(C(C)(C)C)C(C)(C)C)cc1',
    'CataCXium A': 'CCCCP(C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2',
    'None': '',
    'P(Cy)3': 'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1',
    'P(Ph)3': 'c1ccc(P(c2ccccc2)c2ccccc2)cc1',
    'P(o-Tol)3': 'Cc1ccccc1P(c1ccccc1C)c1ccccc1C',
    'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C',
    'SPhos': 'COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
    'XPhos': 'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1',
    'Xantphos': 'CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21',
    'dppf': '[Fe+2].c1ccc(P(c2ccccc2)[c-]2cccc2)cc1',
    'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]',
}
BASES = {
    'CsF': '[Cs+].[F-]',
    'Et3N': 'CCN(CC)CC',
    'K3PO4': '[K+].[K+].[K+].O=P([O-])([O-])[O-]',
    'KOH': '[K+].[OH-]',
    'LiOtBu': '[Li+].CC(C)(C)[O-]',
    'NaHCO3': '[Na+].O=C([O-])O',
    'NaOH': '[Na+].[OH-]',
    'None': '',
}
# Every reaction also contains water and Pd(OAc)2; as constants they carry no
# information and are omitted. The "_V2" entries are variants of the same solvent.
SOLVENTS = {
    'DMF': 'CN(C)C=O',
    'MeCN': 'CC#N',
    'MeOH': 'CO',
    'MeOH/H2O_V2 9:1': 'CO',
    'THF': 'C1CCOC1',
    'THF_V2': 'C1CCOC1',
}
EXPECTED_PRODUCT = 'Cc1ccc2c(cnn2C2CCCCO2)c1-c1ccc2ncccc2c1'


def _lookup(table, name, column):
    key = 'None' if pd.isna(name) else str(name).strip()
    if key not in table:
        raise ValueError(f"Unknown {column} {name!r}; the source data may have changed")
    return table[key]


def convert(xlsx: pd.DataFrame) -> pd.DataFrame:
    """Map the supplementary spreadsheet's rows to ReactionForge columns."""
    rows = []
    for _, r in xlsx.iterrows():
        partner_1 = _lookup(REACTANT_1, r['Reactant_1_Short_Hand'], 'Reactant_1')
        partner_2 = _lookup(REACTANT_2, r['Reactant_2_Name'], 'Reactant_2')
        # One partner carries boron, the other is the (pseudo)halide electrophile
        if 'B' in partner_2.replace('Br', ''):
            electrophile, boron = partner_1, partner_2
        else:
            electrophile, boron = partner_2, partner_1
        rows.append({
            'reactant_smiles': electrophile,
            'boronic_acid_smiles': boron,
            'ligand_smiles': _lookup(LIGANDS, r['Ligand_Short_Hand'], 'ligand'),
            'base_smiles': _lookup(BASES, r['Reagent_1_Short_Hand'], 'base'),
            'solvent_smiles': _lookup(SOLVENTS, r['Solvent_1_Short_Hand'], 'solvent'),
            'yield': float(r['Product_Yield_PCT_Area_UV']),
        })
    df = pd.DataFrame(rows, index=xlsx.index)

    products = {key: suzuki_product(*key)
                for key in set(zip(df['reactant_smiles'], df['boronic_acid_smiles']))}
    df['product_smiles'] = [products[k] for k in zip(df['reactant_smiles'], df['boronic_acid_smiles'])]
    if set(products.values()) != {EXPECTED_PRODUCT}:
        raise ValueError(f"Unexpected products {sorted(set(products.values()))}")
    return df


def _read(source, reader, **kwargs):
    if os.path.exists(source):
        return reader(source, **kwargs)
    print(f"Downloading {source} ...")
    with urllib.request.urlopen(source, timeout=60) as response:
        return reader(io.BytesIO(response.read()), **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Convert the Perera et al. Suzuki dataset')
    parser.add_argument('--xlsx', default=XLSX_URL,
                        help='Path or URL of aap9112_Data_File_S1.xlsx (default: download)')
    parser.add_argument('--split', type=int, default=None,
                        help='Write the standard rxn_yields random split k (0-9) as train/test CSVs')
    parser.add_argument('--split_file', default=None,
                        help='Path or URL of a random_split_k.tsv (default: download for --split)')
    parser.add_argument('--output', default='data/suzuki_perera.csv',
                        help='Output CSV for the whole dataset (without --split)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for train.csv / test.csv (with --split)')
    args = parser.parse_args()

    xlsx = _read(args.xlsx, pd.read_excel)
    if len(xlsx) != 5760:
        sys.exit(f"Expected 5760 reactions, found {len(xlsx)}")
    data = convert(xlsx)

    if args.split is None:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        data.to_csv(args.output, index=False)
        print(f"Wrote {len(data)} reactions -> {args.output}")
        return

    split = _read(args.split_file or SPLIT_URL.format(k=args.split), pd.read_csv,
                  sep='\t', index_col=0)
    # The split file's index is the spreadsheet row; check yields agree
    if (abs(data.loc[split.index, 'yield'].values / 100 - split['y'].values) > 1e-6).any():
        sys.exit("Split file does not match the spreadsheet rows")
    n_train = round(TRAIN_FRACTION * len(split))  # 4032 of 5760
    out_dir = args.output_dir or f'data/suzuki_split{args.split}'
    os.makedirs(out_dir, exist_ok=True)
    data.loc[split.index[:n_train]].to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    data.loc[split.index[n_train:]].to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    print(f"Split {args.split}: {n_train} train / {len(split) - n_train} test -> {out_dir}/")


if __name__ == '__main__':
    main()

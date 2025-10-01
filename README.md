# Overview

[SwiftMHC: A High-Speed Attention Network for MHC-Bound Peptide Identification and 3D Modeling](https://doi.org/10.1101/2025.01.20.633893)

SwiftMHC is a deep learning algorithm for predicting pMHC structure and binding affinity at the same time.
It currently works for HLA-A*0201 9-mers only.

## Speed performance

When running on 1/4 A100 card with batch size 64:
 * binding affinity (BA) prediction takes 0.01 seconds per pMHC case
 * 3D structure prediction without OpenMM (disabled) takes 0.9 seconds per pMHC case.
 * 3D structure prediction with OpenMM takes 2.2 seconds per case.

## Requirements and installation

### Requirements
- Linux only (due to restriction of OpenFold)
- Python ≥3.11
- [PyTorch](https://pytorch.org/get-started/locally/) ≥2.5
    - CUDA is optional, but recommended for training and inference speed.
- [OpenFold](https://github.com/aqlaboratory/openfold)
- [PyMOL](https://pymol.org)

### Installation

#### 1. Install PyTorch
Follow the instructions from PyTorch website https://pytorch.org/get-started/locally/

#### 2. Install OpenFold

```
# Clone OpenFold repository
git clone https://github.com/aqlaboratory/openfold.git

# Enter the OpenFold directory
cd openfold

# Install OpenFold
pip install .

# Install third party dependencies required by OpenFold
scripts/install_third_party_dependencies.sh
```

#### 3. PyMOL

For data preprocessing in SwiftMHC, pymol is required. Download and install it from https://pymol.org

#### 4. Install SwiftMHC

```
# Clone SwiftMHC repository
git clone https://github.com/X-lab-3D/swiftmhc.git

# Enter the SwiftMHC directory
cd swiftmhc

# Install SwiftMHC
pip install .
```

SwiftMHC is now installed.

## Preprocessing data

Preprocessing means to create a file in HDF5 format, containing info in the peptide and MHC protein.
This is only needed if you want to use a new MHC structure or if you want to train a new network.

Preprocessing requires a CSV table in IEDB format. See the data directory for an example.
This table must have the following columns:
- ID (required) : the id under which the row's data will be stored in the HDF5 file. This must correspond to the name of a structure in PDB format.
- allele (required): the name of the MHC allele. (example: HLA-A*02:01) SwiftMHC will use this to identify MHC structures when predicting unlabeled data.
- peptide (optional): the sequence of the peptide. This is used in training, validation, test and not in predicting unlabeled data.
- measurement_value (optional): binding affinity data or classification (BINDING/NONBINDING). This is used in training, validation, test and not in predicting unlabeled data.

Preprocessing requires a reference structure, to align all MHC molecules to.
It also requires a directory containing all the other structures. These may have a peptide in them, but must always contain an MHC structure.

Preprocessing also requires two mask files: a G-domain and a CROSS mask (pocket residues only). See the data directory for examples.
These masks have to be compatible to the reference structure.

To create training, validation, test sets, run:
```
swiftmhc_preprocess IEDB_table.csv ref_mhc.pdb mhcp_binder_models/ \
  mhc_self_attention.mask mhc_cross_attention.mask preprocessed_data.hdf5
```

To preprocess just the MHC allele structures, for predicting unlabeled data, run:
```
swiftmhc_preprocess allele_table.csv ref_mhc.pdb mhc_models/ \
  mhc_self_attention.mask mhc_cross_attention.mask preprocessed_mhcs.hdf5
```

Run `swiftmhc_preprocess --help` for details.

Preprocessing requires data tables, 3D structures and mask files. Check the data directory in this repo for examples.

## Training

This requires preprocessed HDF5 files, containing structures of the MHC protein, peptide and binding affinity or classification data.
We have these available at ...

Run
```
swiftmhc_run -r example train.hdf5 valid.hdf5 test.hdf5
```

Run `swiftmhc_run --help` for details.


This will save the network model to `example/best-predictor.pth`

## Predicting unlabelled data

Do this after training a model (pth format).
Alternatively, there are pretrained models in this repository under the directory named `trained-models`.

Prediction requires preprocessed HDF5 files, containing structures of the MHC protein, for every allele.
The data directory contains a preprocessed hdf5 file for the HLA-A*02:01 allele only.

Prediction also requires a table, linking the peptides to MHC alleles.
It needs to be in CSV format and have the following two columns:
 - a column named 'peptide', holding the sequence of the epitope peptide. Example: LAYYNSCML
 - a column named 'allele', holding the name of the MHC allele. Example: HLA-A*02:01

For example:
To predict unlabeled data, run
```
swiftmhc_predict -B1 trained-models/8k-trained-model.pth data/example-data-table.csv data/HLA-A0201-from-3MRD.hdf5 results/
```

The output `results` directory will contain the BA data and the structures for the peptides that were predicted binding.
The file results/results.csv will hold the BA and class values per MHC,peptide combination.
Note that the affinities in this file are not IC50 or Kd. They correspond to 1 - log_50000(IC50) or 1 - log_50000(Kd).

If the flag --with-energy-minimization is included, SwiftMHC runs OpenMM with an amber99sb/tip3p forcefield to refine the final structure.

Run `swiftmhc_predict --help` for details.

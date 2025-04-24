# Overview

[SwiftMHC: A High-Speed Attention Network for MHC-Bound Peptide Identification and 3D Modeling](https://doi.org/10.1101/2025.01.20.633893)

SwiftMHC is a deep learning algorithm for predicting pMHC structure and binding affinity at the same time.
It currently works for HLA-A*0201 9-mers only.

# Estimated speed

When running on 1/4 A100 card with batch size 64:
 * binding affinity (BA) prediction takes 0.01 seconds per pMHC case
 * 3D structure prediction without OpenMM (disabled) takes 0.9 seconds per pMHC case.
 * 3D structure prediction with OpenMM takes 2.2 seconds per case.

## Dependencies

 - pip3
 - python >= 3.11.5
 - setuptools >= 75.5.0
 - openfold >= 1.0.0
 - position-encoding >= 1.0.0 (github.com/X-lab-3D/position-encoding)
 - PyTorch >= 2.0.1
 - pandas >= 1.5.3
 - numpy >= 1.26.4
 - h5py >= 3.10.0
 - ml-collections >= 0.1.1
 - scikit-learn >= 1.4.1
 - openmm >= 8.1.1 (SwiftMHC needs the cuda version if you were to run on a cuda platform)
 - blosum >= 2.0.3
 - modelcif >= 1.0
 - filelock >= 3.13.1
 - biopython >= 1.8.4
 - PyMol >= 3.1

CUDA is optional

## Installation

First install PyTorch. Follow the instructions from https://pytorch.org/get-started/locally/

Then install openfold, clone this repo: https://github.com/aqlaboratory/openfold
Then from inside that repo, run:

```
pip install -e .

scripts/install_third_party_dependencies.sh
```

For preprocessing, pymol is required. Download and install from https://pymol.org

Then clone the SwiftMHC repo (this repo)
From this repositiry run:
```
pip install -e .
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

Run
```
swiftmhc_predict -B1 trained-models/8k-trained-model.pth table.csv preprocessed_mhcs.hdf5 results/
```

The output `results` directory will contain the BA data and the structures.
The file results/results.csv will hold the BA and class values per MHC,peptide combination.
Note that the affinities in this file are not IC50 or Kd. They correspond to 1 - log_50000(IC50) or 1 - log_50000(Kd).

If the flag --with-energy-minimization is included, SwiftMHC runs OpenMM with an amber99sb/tip3p forcefield to refine the final structure.

Run `swiftmhc_predict --help` for details.

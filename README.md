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

## Inference / Prediction

### Input files

Inference requires the following input files:
- a trained model ([trained-models/8k-trained-model.pth](trained-models/8k-trained-model.pth))
- a CSV file linking the peptides to MHC alleles ([data/example-data-table.csv](data/example-data-table.csv))
- a preprocessed HDF5 file containing MHC structures for every allele ([data/HLA-A0201-from-3MRD.hdf5](data/HLA-A0201-from-3MRD.hdf5))
- an output directory (e.g. `results/`)

The CSV file must have the following columns:
- `peptide` column: holding the sequence of the epitope peptide, e.g. `LAYYNSCML`
- `allele` column: holding the name of the MHC allele, e.g. `HLA-A*02:01`

### Run inference

To run inference, use the command `swiftmhc_predict`. Run `swiftmhc_predict --help` for details.

For example, to predict binding affinity and structure for the peptides in `data/example-data-table.csv` with MHC allele `HLA-A*02:01`, run:
```
swiftmhc_predict --batch-size 1 trained-models/8k-trained-model.pth data/example-data-table.csv data/HLA-A0201-from-3MRD.hdf5 results/
```

The output `results` directory will contain the binding affinity (BA) data and the structures for the peptides that were predicted to bind the MHC.
The file `results/results.csv` will hold the BA and class values per MHC-peptide combination.
Note that the affinities in this file are not IC50 or Kd. They correspond to `1 - log_50000(IC50)` or `1 - log_50000(Kd)`.

If the flag `--with-energy-minimization` is used for the command `swiftmhc_predict`, SwiftMHC will run OpenMM with an amber99sb/tip3p forcefield to refine the final structure.

## Preprocessing data

Preprocessing means to create a file in [HDF5](https://www.hdfgroup.org/solutions/hdf5/) format, containing info in the peptide and MHC protein. This is only needed if you want to use a new MHC structure or train a new network.

### Input files
Preprocessing requires the following files:
- a CSV table in IEDB format ([data/example-data-table.csv](data/example-data-table.csv))
- a reference MHC structure in PDB format ([data/structures/reference-from-3MRD.pdb](data/structures/reference-from-3MRD.pdb))
- a directory containing all other MHC structures in PDB format ([data/structures](data/structures))
- two mask files
    - G-domain mask ([data/HLA-A0201-GDOMAIN.mask](data/HLA-A0201-GDOMAIN.mask))
    - CROSS mask (pocket residue only) ([data/HLA-A0201-CROSS.mask](data/HLA-A0201-CROSS.mask))

The IEDB CSV file must have the following columns:
- `ID` (required) : the id under which the row's data will be stored in the HDF5 file. This must correspond to the name of a structure in PDB format.
- `allele` (required): the name of the MHC allele (e.g. HLA-A*02:01). SwiftMHC will use this to identify MHC structures when predicting unlabeled data.
- `peptide` (optional): the sequence of the peptide. This is used for preprocessing training data, but it's not required when preprocessing an MHC structure for predicting unlabeled data.
- `measurement_value` (optional): binding affinity data or classification (BINDING/NONBINDING). This is used in training, validation and test but not in predicting unlabeled data.

The PDB structures must always contain an MHC structure, and optionally a peptide structure. And the two mask files must be compatible to the reference structure in terms of the number and types of residues.


### Run preprocessing

The preprocessing command is `swiftmhc_preprocess`. Run `swiftmhc_preprocess --help` for details.

To create training, validation and test sets, run:
```
swiftmhc_preprocess IEDB_table.csv ref_mhc.pdb mhcp_binder_models/ \
  mhc_self_attention.mask mhc_cross_attention.mask preprocessed_data.hdf5
```

To preprocess just the MHC allele structures, for predicting unlabeled data, run:
```
swiftmhc_preprocess data/example-data-table.csv \
    data/structures/reference-from-3MRD.pdb \
    data/structures/ \
    data/HLA-A0201-GDOMAIN.mask \
    data/HLA-A0201-CROSS.mask \
    example_preprocessed_data.hdf5
```


## Training

### Input files

Training a new network from scratch requires the following input files:
- a training set in HDF5 format (e.g. `train.hdf5`)
- a validation set in HDF5 format (e.g. `valid.hdf5`)
- a test set in HDF5 format (e.g. `test.hdf5`)

These files can be created using the preprocessing step above.

### Run training

The training command is `swiftmhc_run`. For example:

Run
```
swiftmhc_run -r example train.hdf5 valid.hdf5 test.hdf5
```

This will save the network model to `example/best-predictor.pth`

Run `swiftmhc_run --help` for details and options.

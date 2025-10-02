# Overview

[SwiftMHC: A High-Speed Attention Network for MHC-Bound Peptide Identification and 3D Modeling](https://doi.org/10.1101/2025.01.20.633893)

SwiftMHC is a deep learning algorithm for predicting pMHC structure and binding affinity at the same time.
It currently works for HLA-A*02:01 9-mers only.

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

Inference is the process of predicting binding affinity and optionally a structure of a peptide bound to a major histocompatibility complex (pMHC).

### Input files

Inference requires the following input files:
- a trained model, this repository contains a pre-trained model for 9-mer peptides and the MHC allele HLA-A*02:01: [trained-models/8k-trained-model.pth](trained-models/8k-trained-model.pth)
- a CSV file linking the peptides to MHC alleles, see for example: [data/example-data-table.csv](data/example-data-table.csv)
- a preprocessed HDF5 file containing MHC structures for every allele. For the allele HLA-A*02:01, such a file is pre-made and available at: [data/HLA-A0201-from-3MRD.hdf5](data/HLA-A0201-from-3MRD.hdf5)

The input CSV file must have the following columns:
- `peptide` column: holding the sequence of the epitope peptide, e.g. `LAYYNSCML`
- `allele` column: holding the name of the MHC allele, e.g. `HLA-A*02:01`

### Run inference

To run inference, use the command `swiftmhc_predict`. Run `swiftmhc_predict --help` for details.

For example, to predict binding affinity and structure for the peptides in `data/example-data-table.csv` with MHC allele `HLA-A*02:01`, run:
```
swiftmhc_predict --batch-size 1 --num-builders 1 trained-models/8k-trained-model.pth data/example-data-table.csv data/HLA-A0201-from-3MRD.hdf5 results/
```

Here, `results` must be a directory. If this directory doesn't exist it will be created.
The output `results` directory will contain the binding affinity (BA) data and the structures for the peptides that were predicted to bind the MHC.
The file `results/results.csv` will hold the BA and class values per MHC-peptide combination.
Note that the affinities in this file are not IC50 or Kd. They correspond to `1 - log_50000(IC50)` or `1 - log_50000(Kd)`.

If the flag `--with-energy-minimization` is used for the command `swiftmhc_predict`, SwiftMHC will run OpenMM with an amber99sb/tip3p forcefield to refine the final structure.

To predict just the binding affinity without a structure. Run with no builders:
```
swiftmhc_predict --num-builders 0 --batch-size 1 trained-models/8k-trained-model.pth data/example-data-table.csv data/HLA-A0201-from-3MRD.hdf5 results/
```

## Preprocessing data

Preprocessing is the process of creating a file in [HDF5](https://www.hdfgroup.org/solutions/hdf5/) format, containing info in the peptide and MHC protein.
This is only needed if you want to use a new MHC structure for **inference** or train a new network.

There are two ways of preprocessing:
 1. Preprocessing training datasets. This requires that structures are provided that contain the MHC (as PDB chain M) and the peptide (as PDB chain P).
    Binding affinity data must also be provided, as this information is used in the training process.

 2. Preprocessing MHC structures for inference. In this case only the MHC structures must be provided as PDB chain M.

In both cases a reference structure must be provided, so that all the structures can be aligned and superposed to it. This is done using pymol.
A reference structure is available in this repository: [data/structures/reference-from-3MRD.pdb](data/structures/reference-from-3MRD.pdb).
This reference structure has been tested to work with HLA-A*02:01 structures, but other alleles likely work as well.

Additionally to the reference structure, two mask files must be provided: one mask file specifies which residues are considered G-domain residues and the other specifies which residues are groove residues that lie close to the peptide.
The G-domain mask is used for MHC self attention and the groove residues are used for cross attention between the peptide and the MHC.
For the reference structure [data/structures/reference-from-3MRD.pdb](data/structures/reference-from-3MRD.pdb), two mask files are available in this repository:
 - [data/HLA-A0201-GDOMAIN.mask](data/HLA-A0201-GDOMAIN.mask)
 - [data/HLA-A0201-CROSS.mask](data/HLA-A0201-CROSS.mask)

### Preprocessing training datasets

#### Input files

Preprocessing training data requires the following files:
 - CSV table in IEDB format, see for an example: [data/example-data-table.csv](data/example-data-table.csv)
 - a reference MHC structure in PDB format, for example: [data/structures/reference-from-3MRD.pdb](data/structures/reference-from-3MRD.pdb)
 - a directory containing all training pMHC structures in PDB format. The contents of this directory must be PDB files that are named corresponding to the ID column in the CSV table.
   Furthermore the PDB files must have the extension .pdb. For example: `BA-74141.pdb`, where BA-74141 corresponds to an ID in the table.
 - two mask files: G-domain and CROSS, containing residues corresponding to residues in the reference MHC structure.
   For example: [data/HLA-A0201-GDOMAIN.mask](data/HLA-A0201-GDOMAIN.mask) and [data/HLA-A0201-CROSS.mask](data/HLA-A0201-CROSS.mask)

For preprocessing training datasets, the IEDB CSV file must have the following columns:
 - `ID`: the id under which the row's data will be stored in the HDF5 file. This must correspond to the name of a structure in PDB format.
 - `allele`: the name of the MHC allele (e.g. HLA-A*02:01). SwiftMHC will use this to identify MHC structures when predicting unlabeled data.
 - `peptide`: the amino acid sequence of the peptide.
 - `measurement_value`: binding affinity data or classification (BINDING/NONBINDING).

#### Run preprocessing for training datasets

For preprocessing training datasets, lets take example data from DOI: https://doi.org/10.5281/zenodo.14968655.
From the compressed tar file we use the following:
 - a CSV table in IEDB format: `input-data/IEDB-BA-with-clusters.csv`. It has the required columns, but it also contains cluster ids so that the data can be separated by cluster.
 - PANDORA models, representing pMHC structures: `input-data/swiftmhc/pandora-models-for-training-swiftmhc/`.

The preprocessing command is `swiftmhc_preprocess`. Run `swiftmhc_preprocess --help` for details.

To preprocess training data, in 32 simultaneous processes, run:

```
swiftmhc_preprocess /path/to/extracted/input-data/IEDB-BA-with-clusters.csv \
                    data/structures/reference-from-3MRD.pdb \
                    /path/to/extracted/input-data/swiftmhc/pandora-models-for-training-swiftmhc/ \
                    data/HLA-A0201-GDOMAIN.mask \
                    data/HLA-A0201-CROSS.mask \
                    example_preprocessed_training_data.hdf5
                    --processes 32
```

This will generate a HDF5 file `example_preprocessed_training_data.hdf5`, that can be used for training.

### Preprocessing MHC structures for inference

#### Input files

Preprocessing MHC structures for inference requires the following files:

 - CSV table, with columns: `ID` and `allele`. The ID column will contain the identifiers under which the MHC structures will be stored in the HDF5 file.
   The allele column must contain allele names, which will be used to look up the MHC structure in the HDF5 file during inference.
 - a reference MHC structure in PDB format, for example: [data/structures/reference-from-3MRD.pdb](data/structures/reference-from-3MRD.pdb)
 - a directory containing all MHC structures in PDB format. The contents of this directory must be PDB files that are named corresponding to the allele column in the CSV table.
   Furthermore the PDB files must have the extension .pdb. For example: `HLA-A0201.pdb`, where HLA-A0201 corresponde to a name under the allele column.
 - two mask files: G-domain and CROSS, containing residues corresponding to residues in the reference MHC structure.
   For example: [data/HLA-A0201-GDOMAIN.mask](data/HLA-A0201-GDOMAIN.mask) and [data/HLA-A0201-CROSS.mask](data/HLA-A0201-CROSS.mask)

#### Run preprocessing MHC structures for inference

For preprocessing MHC structures for inference. Let's take the the HLA-A*02:01 structure [data/structures/reference-from-3MRD.pdb](data/structures/reference-from-3MRD.pdb) in this repo as an example.

First copy the pdb file to an assigned directory and create the input table:

```
mkdir preprocess-HLA-A0201
cp data/structures/reference-from-3MRD.pdb preprocess-HLA-A0201/HLA-A0201.pdb

echo ID,allele > preprocess-HLA-A0201.csv
echo HLA-A0201,HLA-A0201 >> preprocess-HLA-A0201.csv
```

Then, to preprocess the MHC structure, run:

```
swiftmhc_preprocess preprocess-HLA-A0201.csv \
                    data/structures/reference-from-3MRD.pdb \
                    preprocess-HLA-A0201 \
                    data/HLA-A0201-GDOMAIN.mask \
                    data/HLA-A0201-CROSS.mask \
                    preprocessed-HLA-A0201.hdf5
```

This will generate a HDF5 file `preprocessed-HLA-A0201.hdf5`, that can be used for predicting pMHC structures and binding affinities on the HLA-A*02:01 allele.

## Training

### Input files

Training a new network from scratch requires the following input files:
- a training set in HDF5 format (e.g. `train.hdf5`)
- a validation set in HDF5 format (e.g. `valid.hdf5`), this is optional
- a test set in HDF5 format (e.g. `test.hdf5`), this is also optional

These files can be created using the preprocessing step above.

### Run training

For preprocessing training datasets, lets take example data from DOI: https://doi.org/10.5281/zenodo.14968655.
From the compressed tar file we use the following:
 - A training HDF5 dataset: `preprocessed/train_fold0.hdf5`
 - A validation HDF5 dataset: `preprocessed/valid_fold0.hdf5`
 - A test HDF5 dataset: `preprocessed/BA_cluster0.hdf5`

We will train based on a 10-fold cross validation. `preprocessed/BA_cluster0.hdf5` contains the data from cluster 0. The data from the remaining 9 clusters is in `preprocessed/train_fold0.hdf5` (90%) and `preprocessed/valid_fold0.hdf5` (10%).
The dataset `preprocessed/train_fold0.hdf5` will be used for training and the dataset `preprocessed/valid_fold0.hdf5` will be used for selecting the best model.

To perform training and evaluation, run:

```
swiftmhc_run -r example /path/to/extracted/preprocessed/train_fold0.hdf5 /path/to/extracted/preprocessed/valid_fold0.hdf5 /path/to/extracted/preprocessed/BA_cluster0.hdf5
```

This will save the network model to `example/best-predictor.pth`

Run `swiftmhc_run --help` for details and options.

### Run evaluation on a pretrained model

Evaluation runs the same script as training, but it doesn't alter the network model.
The difference between evaluation and inference is that evaluation is done using a separate X-ray structure or homology model for
every pMHC combination, where inference uses the same MHC structure for every pMHC on a particular allele.
Evaluation is typically used for cross validation.

To do the evaluation on a pretrained model, run:

```
swiftmhc_run -p /path/to/extracted/network-models/swiftmhc/swiftmhc-default/model-for-fold-0.pth -t /path/to/extracted/preprocessed/BA_cluster0.hdf5 --num-builders 1 -r evaluation
```

This will output binding affinity data to `evaluation/BA_cluster0-affinities.csv`
and it will output all structures to a single file in compressed format: `evaluation/BA_cluster0-predicted.hdf5`.

To extract the output structures from the HDF5 file, run:

```
swiftmhc_hdf5_to_pdb evaluation/BA_cluster0-predicted.hdf5
```

This will output all PDB files to a directory named `evaluation/BA_cluster0-predicted`.

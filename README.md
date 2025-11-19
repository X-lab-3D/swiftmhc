# Overview

[SwiftMHC: A High-Speed Attention Network for MHC-Bound Peptide Identification and 3D Modeling](https://doi.org/10.1101/2025.01.20.633893)

SwiftMHC is a deep learning algorithm for predicting pMHC structure and binding affinity at the same time.
It currently works for `HLA-A*02:01` 9-mers only.

## Speed performance

When running on one A100 card with batch size 64:
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

#### 3. Install PyMOL

PyMOL is required for data preprocessing in SwiftMHC. You can install the open-source version via conda:

```
conda install -c conda-forge pymol-open-source
```

To install the proprietary version of PyMOL, please refer to the instructions on the website https://pymol.org.

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
- a trained model, this repository contains a pre-trained model for 9-mer peptides and the MHC allele `HLA-A*02:01`: [trained-models/8k-trained-model.pth](trained-models/8k-trained-model.pth)
- a CSV file linking the peptides to MHC alleles, see for example: [data/example-inference-data-table.csv](data/example-inference-data-table.csv)
- a preprocessed HDF5 file containing MHC structures for every allele. For the allele `HLA-A*02:01`, such a file is pre-made and available at: [data/HLA-A0201-from-3MRD.hdf5](data/HLA-A0201-from-3MRD.hdf5)

The input CSV file must have the following columns:
- `peptide` column: holding the sequence of the epitope peptide, e.g. `LAYYNSCML`
- `allele` column: holding the name of the MHC allele, e.g. `HLA-A*02:01`

### Run inference

To run inference, use the command `swiftmhc_predict`. Run `swiftmhc_predict --help` for details.

For example, to predict binding affinity and structure for the peptides in `data/example-inference-data-table.csv` with MHC allele `HLA-A*02:01`, run:
```
swiftmhc_predict --num-builders 1 \
    --batch-size 1 \
    trained-models/8k-trained-model.pth \
    data/example-inference-data-table.csv \
    data/HLA-A0201-from-3MRD.hdf5 \
    results
```

Here, `results` must be a directory. If this directory doesn't exist it will be created.
The output `results` directory will contain the binding affinity (BA) data and the structures for the peptides that were predicted to bind the MHC.
The file `results/results.csv` will hold the BA and class values per MHC-peptide combination.
Note that the affinities in this file are not IC50 or Kd. They correspond to `1 - log_50000(IC50)` or `1 - log_50000(Kd)`.

If the flag `--with-energy-minimization` is used for the command `swiftmhc_predict`, SwiftMHC will run OpenMM with an amber99sb/tip3p forcefield to refine the final structure.

To predict just the binding affinity without a structure, set `--num-builders` to 0, for example:
```
swiftmhc_predict --num-builders 0 \
    --batch-size 1 \
    trained-models/8k-trained-model.pth \
    data/example-inference-data-table.csv \
    data/HLA-A0201-from-3MRD.hdf5 \
    results
```

## Preprocessing data

Preprocessing is the process of creating a file in [HDF5](https://www.hdfgroup.org/solutions/hdf5/) format, containing info in the peptide and MHC protein.
This is only needed if you want to 1) use a new MHC structure for inference or 2) retrain SwiftMHC.

Three required input files are already included in this repository:

A **reference structure** is provided at: [data/structures/reference-from-3MRD.pdb](data/structures/reference-from-3MRD.pdb)

This structure has been tested with `HLA-A*02:01` but is likely suitable for other alleles as well. It is used to align and superpose all training structures (using PyMOL).

In addition, two **mask files** are provided, which define the residues used for different attention mechanisms:
 - **MHC G-domain residues:** [data/HLA-A0201-GDOMAIN.mask](data/HLA-A0201-GDOMAIN.mask)
 - **CROSS residues (MHC groove residues):** [data/HLA-A0201-CROSS.mask](data/HLA-A0201-CROSS.mask)

 These files correspond to the provided reference structure and are ready to use.

### Preprocessing training datasets

If users want to retrain SwiftMHC on new training data, they need to run  preprocessing on the training data before running the training script. This section shows how.

#### Input files

The following input files are required:

1. **CSV table containing the binding affinity data in IEDB format**
   - Example: [`data/example-training-data-table.csv`](data/example-training-data-table.csv)

     For preprocessing training datasets, the data CSV file must have the following columns:
 - `ID`: the id under which the row's data will be stored in the HDF5 file. This must correspond to the PDB file name (see below).
 - `allele`: the name of the MHC allele (e.g. `HLA-A*02:01`). This is added to the data for administrative puposes.
 - `peptide`: the amino acid sequence of the peptide.
 - `measurement_value`: binding affinity data (IC50 or Kd in nM) or classification (BINDING/NONBINDING as string).


2. **Directory of training pMHC structures (PDB format)**
   - The directory must contain PDB files named according to the **ID** column in the CSV table.
   - Filenames must end with `.pdb` (e.g., `BA-74141.pdb`, where `BA-74141` is an ID from the CSV table).
   - Each PDB file must contain:
     - the **MHC molecule** as chain **M**
     - the **peptide** as chain **P**
   - If your PDB files use different chain IDs, we recommend adjusting them using [pdb-tools](https://www.bonvinlab.org/pdb-tools/).
     Example:
     ```bash
     python pdb_chain.py -M 1AKJ.pdb
     ```


#### Run preprocessing for training datasets

For preprocessing training datasets, let's take example data from DOI: https://doi.org/10.5281/zenodo.14968655.
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

This process generates an HDF5 file `example_preprocessed_training_data.hdf5` which can be used for model training.
The file contains structural information for both the peptide and the MHC, including the proximity matrix, transformation matrices, atom positions, amino acid types, and torsion angles.
In addition, binding affinity values are stored for each pMHC complex.

### Preprocessing MHC structures for inference

In case during the inference the user wants to use a different MHC structure other than what we provided  [data/HLA-A0201-from-3MRD.hdf5](data/HLA-A0201-from-3MRD.hdf5), they can generate a preprocessed hdf5 for their MHC structure. This section shows how.

#### Input files

Only the MHC structures (with chain ID as **M**) are required.

Preprocessing MHC structures for inference requires the following files:

 - CSV table, with columns: `ID` and `allele`. The ID column will contain the user-defined identifiers, under which the MHC structures will be stored in the HDF5 file.
   The allele column must contain allele names, which will be stored in the HDF5 file to be used to look up the MHC structure in the HDF5 file during inference.

 - a directory containing all MHC structures in PDB format. The contents of this directory must be PDB files that are named corresponding to the allele column in the CSV table.
   Furthermore the PDB files must have the extension .pdb. For example: `HLA-A0201.pdb`, where HLA-A0201 corresponde to a name under the allele column.


#### Run preprocessing MHC structures for inference

Let's take the the `HLA-A*02:01` structure [data/structures/example-preprocess-HLA-A0201/HLA-A0201.pdb](data/structures/example-preprocess-HLA-A0201/HLA-A0201.pdb) in this repo as an example.

Then, to preprocess the MHC structure, run:

```
swiftmhc_preprocess data/example-preprocess-HLA-A0201.csv \
                    data/structures/reference-from-3MRD.pdb \
                    data/structures/example-preprocess-HLA-A0201/ \
                    data/HLA-A0201-GDOMAIN.mask \
                    data/HLA-A0201-CROSS.mask \
                    example-preprocessed-HLA-A0201.hdf5
```

This will generate a HDF5 file `preprocessed-HLA-A0201.hdf5`, that can be used for predicting pMHC structures and binding affinities on the user-specified `HLA-A*02:01` structure.

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
swiftmhc_run -r example \
    /path/to/extracted/preprocessed/train_fold0.hdf5 \
    /path/to/extracted/preprocessed/valid_fold0.hdf5 \
    /path/to/extracted/preprocessed/BA_cluster0.hdf5
```

This will save the network model to `example/best-predictor.pth`

Run `swiftmhc_run --help` for details and options.

### Run evaluation on a pretrained model

Evaluation runs the same script as training but does not update model weights. Unlike inference, which uses a single MHC structure per allele and no target values, evaluation uses  one HDF5 file that includes target values (e.g., peptide structures, binding affinities) for comparison only, not as inputs. This part was used to generate the cross-validation results in the article.

To do the evaluation on a pretrained model, run:

```
swiftmhc_run -t -r evaluation \
    -p /path/to/extracted/network-models/swiftmhc/swiftmhc-default/model-for-fold-0.pth \
    /path/to/extracted/preprocessed/BA_cluster0.hdf5
```

This will output binding affinity data to `evaluation/BA_cluster0-affinities.csv`
and it will output all structures to a single file in compressed format: `evaluation/BA_cluster0-predicted.hdf5`.

To extract the output structures from the HDF5 file, run:

```
swiftmhc_hdf5_to_pdb evaluation/BA_cluster0-predicted.hdf5
```

This will output all PDB files to a directory named `evaluation/BA_cluster0-predicted`.

## Data Evaluation

### Point Mutations

The script `scripts/evaluation/find_mutations.py` can be used to identify single point mutations in IEDB CSV tables.
It will create an output file, containing data on the wild type (wt) and mutant (mut).
The script `scripts/evaluation/get_ddG.py` can be used to calculate the corresponding energy changes (ΔΔG) from the predicted and true affinity values

### Structures

On X-ray or predicted structures:
The script `scripts/evaluation/measure_clashes.py` can be used to find clashes.
The script `scripts/evaluation/measure_chirality.py` evaluates whether the amino acids are L or D form.
The script `scripts/evaluation/measure_omega_angles.py` measures backbone ω angles.
The script `scripts/evaluation/measure_ramachandran_angles.py` measures backbone φ and ψ angles.

To compare Cα positions between a predicted and X-ray structure, RMSD can be calculated using the script `scripts/evaluation/rmsd-calpha-mhc-9mer-peptides.bash`.
This script will use profit to superpose the MHC structures and output the RMSD between the peptides.

### Finding overlap between Datasets.

To find the overlap between MHCfold & AlphaFold2-FineTune training sets and a test set, use the script `scripts/evaluation/find_train_test_overlap.py`.
This requires a copy of the RCSB PDB database [https://www.wwpdb.org/ftp/pdb-ftp-sites](https://www.wwpdb.org/ftp/pdb-ftp-sites),
a PANDORA database [https://github.com/x-lab-3D/PANDORA](https://github.com/x-lab-3D/PANDORA) and
a copy of the AlphaFold2-FineTune pMHC templates dataset [https://files.ipd.uw.edu/pub/alphafold_finetune_motmaen_pnas_2023/datasets_alphafold_finetune_v2_2023-02-20.tgz](https://files.ipd.uw.edu/pub/alphafold_finetune_motmaen_pnas_2023/datasets_alphafold_finetune_v2_2023-02-20.tgz).

### BA avaluation

The script `scripts/evaluation/auc.py` can be used to calculate the AUC values for assessing BA prediction quality.
It requires one table, holding the SwiftMHC `output affinity` and the ground truth binary value `true class` (binding=1, non binding=0).

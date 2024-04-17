# SwiftMHC

A deep learning algorithm for predicting pMHC structure and binding affinity at the same time.

## DEPENDENCIES

 - python >= 3.11.5
 - openfold >= 1.0.0
 - pytorch >= 2.0.1
 - pandas >= 1.5.3
 - numpy >= 1.26.4
 - h5py >= 3.10.0
 - ml-collections >= 0.1.1
 - scikit-learn >= 1.4.1
 - openmm >= 8.1.1

## INSTALL

Run:
```
python setup.py install
```

## PREPROCESSING DATA

Preprocessing means to create a file in HDF5 format, containing info in the peptide and MHC protein.
It requires the following:
 - structures of the MHC molecules and optionally peptides for training
 - a reference structure, to align all MHC molecules to
 - two residue masks, compatible to the reference structure, one for self-attention (G-domain) and one for cross attention (pocket)
 - optionally, binding affinity data or classification (BINDING/NONBINDING), for training

To create training, validation, test sets, run:
```
swiftmhc_preprocess IEdb_table.csv ref_mhc.pdb mhcp_binder_models/ mhc_self_attention.mask mhc_cross_attention.mask preprocessed_data.hdf5
```

To preprocess just the MHC allele structures, for predicting unlabeled data, run:
```
swiftmhc_preprocess allele_table.csv ref_mhc.pdb mhc_models/ mhc_self_attention.mask mhc_cross_attention.mask preprocessed_mhcs.hdf5
```

Run `swiftmhc_preprocess --help` for details.


## TRAINING

This requires preprocessed HDF5 files, containing structures of the MHC protein, peptide and binding affinity or classification data.

Run
```
swiftmhc_run train.hdf5 valid.hdf5 test.hdf5
```

Run `swiftmhc_run --help` for details.


This will save the model to `best-predictor.pth`

## PREDICTING UNLABELED DATA

This requires preprocessed HDF5 files, containing structures of the MHC protein, for every allele.
It also requires a table, linking the peptides to MHC alleles.

Run
```
swiftmhc_predict best-predictor.pth table.csv preprocessed_mhcs.hdf5 results/
```

Run `swiftmhc_predict --help` for details.

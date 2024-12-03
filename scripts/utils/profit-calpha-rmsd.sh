#!/bin/sh

# This is the script for measuring the C-alpha RMSD for peptide-MHC complexes.
# It requires profit: http://www.bioinf.org.uk/software/profit/

# Precondition is that the target and mobile structures have the same filenames and numbering
# and that the MHC chain is named 'M' and the peptide chain is named 'P'.
# If this is not the case, then this script can be adapted accordingly.

# Input to this script are two directories: the directory with the target PDB files and the directory with the mobile PDB files.

# This script outputs the profit text. The first number in the output text is the MHC C-alpha RMSD. The second number is the peptide C-alpha RMSD.

if [ $# -ne 2 ] ; then
  echo Usage: $0 target_dir mobile_dir
  exit 0
fi

TARGET_DIR=$1
MOBILE_DIR=$2

for mob in $(ls $MOBILE_DIR | grep '\.pdb$') ; do

  pdb=$(basename $mob)
  ref=$TARGET_DIR/$pdb
  mob=$MOBILE_DIR/$mob

  if ! [ -f $ref ] ; then
    echo missing $ref > /dev/stderr
    exit 1
  fi

  id=$(basename $pdb)
  echo "$id chain M (first) and chain P (second)":

  profit << EOF
    ref $ref
    mob $mob
    atom CA
    zone M3-M179:M3-M179
    fit
    rzone P1-P9:P1-P9
    quit
EOF
done


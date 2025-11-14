#!/bin/sh

# this script requires profit to work.

if [ $# -ne 2 ] ; then
    echo "Usage: $0 re:wqference_directory mobile_directory"
    exit 0
fi

REF_DIR=$1

MOB_DIR=$2


for mob in $(ls $MOB_DIR | grep '\.pdb$') ; do

  pdb=$(basename $mob)
  ref=$REF_DIR/$pdb
  mob=$MOB_DIR/$mob

  if ! [ -f $ref ] ; then
    echo missing $ref > /dev/stderr
    exit 1
  fi

  id=$(basename $pdb)
  id=${id%.pdb}
  echo $id chains M P

  # the script processes pairs of PDB files with the same name
  # output is three per pair
  #   the first line states the PDB file's name
  #   the second line states the MHC RMSD from fitting the two structures
  #   the third line states the RMSD for the 9-mer peptide

  profit << EOF | grep RMS:
    ref $ref
    mob $mob
    atom CA
    zone M3-M179:M3-M179
    fit
    rzone P1-P9:P1-P9
    quit
EOF
done


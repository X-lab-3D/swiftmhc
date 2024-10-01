from torch.nn.functional import one_hot
import torch

from openfold.np.residue_constants import restypes as openfold_one_letter_order

from ..models.amino_acid import AminoAcid


AMINO_ACID_DIMENSION = 22

# data for the known amino acids
alanine = AminoAcid("alanine", "ALA", "A")
cysteine = AminoAcid("cysteine", "CYS", "C")
aspartate = AminoAcid("aspartate", "ASP", "D")
glutamate = AminoAcid("glutamate", "GLU", "E")
phenylalanine = AminoAcid("phenylalanine", "PHE", "F")
glycine = AminoAcid("glycine", "GLY", "G")
histidine = AminoAcid("histidine", "HIS", "H")
isoleucine = AminoAcid("isoleucine", "ILE", "I")
leucine = AminoAcid("leucine", "LEU", "L")
methionine = AminoAcid("methionine", "MET", "M")
asparagine = AminoAcid("asparagine", "ASN", "N")
pyrolysine = AminoAcid("pyrolysine", "PYL", "O")
glutamine = AminoAcid("glutamine", "GLN", "Q")
arginine = AminoAcid("arginine", "ARG", "R")
serine = AminoAcid("serine", "SER", "S")
threonine = AminoAcid("threonine", "THR", "T")
selenocysteine = AminoAcid("selenocysteine", "SEC", "U")
valine = AminoAcid("valine", "VAL", "V")
tyrosine = AminoAcid("tyrosine", "TYR", "Y")
tryptophan = AminoAcid("tryptophan", "TRP", "W")
lysine = AminoAcid("lysine", "LYS", "K")
proline = AminoAcid("proline", "PRO", "P")

unknown_amino_acid = AminoAcid("unknown", "UNK", "X")


# lists, referrring to amino acid objects
canonical_amino_acids = [alanine, cysteine, aspartate, glutamate, phenylalanine,
                         glycine, histidine, isoleucine, leucine, methionine,
                         asparagine, glutamine, arginine, serine, threonine,
                         valine, tyrosine, tryptophan, lysine, proline]

all_amino_acids = [alanine, cysteine, aspartate, glutamate, phenylalanine,
                   glycine, histidine, isoleucine, lysine, leucine, methionine,
                   asparagine, pyrolysine, glutamine, arginine, serine, threonine,
                   selenocysteine, valine, tyrosine, tryptophan, lysine, proline]

# dictionaries, for fast lookup
amino_acids_by_name = {amino_acid.name: amino_acid for amino_acid in all_amino_acids}
amino_acids_by_code = {amino_acid.three_letter_code: amino_acid for amino_acid in all_amino_acids}
amino_acids_by_letter = {amino_acid.one_letter_code: amino_acid for amino_acid in all_amino_acids}

# Encode the amino acids in the same way as openfold does:
amino_acids_by_one_hot_index = {}
for index, one_letter_code in enumerate(openfold_one_letter_order):
    amino_acid = amino_acids_by_letter[one_letter_code]
    amino_acid.one_hot_code = one_hot(torch.tensor(index), AMINO_ACID_DIMENSION).to(dtype=torch.bfloat16)
    amino_acid.index = index

    amino_acids_by_one_hot_index[index] = amino_acid

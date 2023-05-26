from typing import List, Union

import torch

from ..models.amino_acid import AminoAcid
from ..domain.amino_acid import amino_acids_by_one_hot_index, unknown_amino_acid


def one_hot_decode_sequence(encoded_sequence: torch.Tensor) -> List[Union[AminoAcid, None]]:
    """ 

    Args:
        encoded_sequence: [sequence_length, AMINO_ACID_DIMENSION]

    Returns: a list of amino acids where None means gap
    """

    sequence_length = encoded_sequence.shape[0]

    amino_acids = []
    for residue_index in range(sequence_length):

        one_hot_code = encoded_sequence[residue_index]

        if torch.all(one_hot_code == 0.0):

            amino_acid = None

        else:
            one_hots = torch.nonzero(one_hot_code)

            if all([dimension == 1 for dimension in one_hots.shape]):

                one_hot_index = one_hots.item()

                amino_acid = amino_acids_by_one_hot_index[one_hot_index]

            else:  # not a one hot code

                amino_acid = unknown_amino_acid

        amino_acids.append(amino_acid)

    return amino_acids


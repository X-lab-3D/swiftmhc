from math import floor
from typing import List
from typing import Tuple
from typing import Union
import torch
from ..domain.amino_acid import amino_acids_by_one_hot_index
from ..domain.amino_acid import unknown_amino_acid
from ..models.amino_acid import AminoAcid


def one_hot_decode_sequence(encoded_sequence: torch.Tensor) -> List[Union[AminoAcid, None]]:
    """Args:
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


def stretch_sequence(
    encoded_sequence: torch.Tensor, new_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes a sequence as a masked sequence with given length.
    XXXXXXXXX becomes XXXX----X---XXXX

    Args:
        encoded_sequence [L, D]
        new_length(N)

    Returns:
        stretched sequence [N, D]
        mask [N]
    """
    if encoded_sequence.shape[0] < 8:
        raise ValueError(
            f"sequence of length {encoded_sequence.shape[0]} is too short, at least 8 required"
        )

    center_i = int(floor(new_length / 2))
    center_length = encoded_sequence.shape[0] - 8
    center_nlength = int(floor(center_length / 2))
    center_clength = center_length - center_nlength

    stretched_sequence = encoded_sequence.new_zeros([new_length] + list(encoded_sequence.shape[1:]))
    stretched_sequence[:4] = encoded_sequence[:4]
    stretched_sequence[center_i - center_nlength : center_i + center_clength] = encoded_sequence[
        4:-4
    ]
    stretched_sequence[-4:] = encoded_sequence[-4:]

    mask = encoded_sequence.new_zeros([new_length], dtype=torch.bool)
    mask[:4] = True
    mask[center_i - center_nlength : center_i + center_clength] = True
    mask[-4:] = True

    return stretched_sequence, mask

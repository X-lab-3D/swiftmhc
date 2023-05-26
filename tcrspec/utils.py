from typing import List, Union

import torch

from .models.amino_acid import AminoAcid
from .domain.amino_acid import amino_acids_by_one_hot_index, unknown_amino_acid


def zero_pad(input_: torch.Tensor, dimension_index: int, new_size: int) -> torch.Tensor:

    "pads the given dimension with zeros and returns the new tensor"

    if new_size == input_.shape[dimension_index]:
        return input_

    elif new_size < input_.shape[dimension_index]:
        raise ValueError(f"new size {new_size} is too small for tensor dimension {dimension_index} of size {input_.shape[dimension_index]}")

    pad_size = new_size - input_.shape[dimension_index]

    pad_shape = list(input_.shape[:dimension_index]) + [pad_size] + list(input_.shape[dimension_index + 1:])

    pad = torch.zeros(pad_shape).to(input_.device)

    return torch.cat((input_, pad), dim=dimension_index)


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


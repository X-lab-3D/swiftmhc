import torch
from openfold.np.residue_constants import restypes
from swiftmhc.dataset import ProteinLoopDataset


def test_dataset():
    protein_maxlen = 200

    dataset = ProteinLoopDataset(
        "tests/data/data.hdf5", torch.device("cpu"), torch.float32, 16, protein_maxlen
    )
    i = dataset.entry_names.index("BA-99998")

    peptide_sequence = "".join(
        [restypes[i] for i in dataset[i]["peptide_sequence_onehot"].nonzero(as_tuple=True)[1]]
    )
    assert peptide_sequence == "YLLGDSDSVA"

    protein_sequence = "".join(
        [restypes[i] for i in dataset[i]["protein_sequence_onehot"].nonzero(as_tuple=True)[1]]
    )
    assert (
        protein_sequence
        == "SHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASRRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTLQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQWRAYLEGTCVEWLRRYLENGKETLQR"
    )

    rigid_tensor = dataset[i]["protein_backbone_rigid_tensor"]
    assert rigid_tensor.shape == (protein_maxlen, 4, 4)

    assert (rigid_tensor[0] != rigid_tensor[1]).any()

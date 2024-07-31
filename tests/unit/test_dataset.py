import torch
from openfold.np.residue_constants import restypes

from swiftmhc.dataset import ProteinLoopDataset


def test_dataset():

    dataset = ProteinLoopDataset("tests/data/data.hdf5", torch.device("cpu"), 16, 200)
    i = dataset.entry_names.index("BA-99998")

    peptide_sequence = ''.join([restypes[i] for i in dataset[i]["peptide_sequence_onehot"].nonzero(as_tuple=True)[1]])
    assert peptide_sequence == "YLLGDSDSVA"

    protein_sequence = ''.join([restypes[i] for i in dataset[i]["protein_sequence_onehot"].nonzero(as_tuple=True)[1]])
    assert protein_sequence == "SHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASRRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTLQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQWRAYLEGTCVEWLRRYLENGKETLQR"


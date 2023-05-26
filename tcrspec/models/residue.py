from ..models.amino_acid import AminoAcid


class Residue:
    def __init__(self, id_, amino_acid: AminoAcid):
        self.id = id_
        self.amino_acid = amino_acid
        self.atoms = {}
        self.mask = False

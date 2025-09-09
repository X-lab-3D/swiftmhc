class AminoAcid:
    """Holds data for amino acids"""

    def __init__(self, name: str, three_letter_code: str, one_letter_code: str):
        self.name = name
        self.three_letter_code = three_letter_code
        self.one_letter_code = one_letter_code

        self.one_hot_code = None

    def __repr__(self):
        return self.three_letter_code

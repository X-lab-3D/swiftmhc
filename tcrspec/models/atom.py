import torch


class Atom:
    def __init__(self, element: str, position: torch.Tensor, occupancy: float):
        """
        Args:
            position is a tensor of 3 floats: x, y ,z
        """

        self.element = element
        self.position = position
        self.occupancy = occupancy

from enum import Enum


class ComplexClass(Enum):
    """A value to be assigned in classification prediction"""

    NONBINDING = 0
    BINDING = 1

    @staticmethod
    def from_string(s: str):
        if s.upper() == "NONBINDING":
            return ComplexClass.NONBINDING

        elif s.upper() == "NEGATIVE":
            return ComplexClass.NONBINDING

        elif s.upper() == "BINDING":
            return ComplexClass.BINDING

        elif s.upper() == "POSITIVE":
            return ComplexClass.BINDING

        raise ValueError(s)

    @staticmethod
    def from_int(i: int):
        if i > 0:
            return ComplexClass.BINDING
        else:
            return ComplexClass.NONBINDING

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

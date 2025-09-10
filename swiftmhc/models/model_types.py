from enum import Enum


class ModelType(Enum):
    """Defines whether the model does regression or classification."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
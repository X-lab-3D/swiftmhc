import ml_collections as mlc

from .models.types import ModelType

config = mlc.ConfigDict(
    {
        "c_s": 32,
        "c_z": 1,
        "c_hidden": 16,
        "c_resnet": 64,
        "c_transition": 64,
        "no_heads": 2,
        "no_qk_points": 4,
        "no_v_points": 8,
        "dropout_rate": 0.1,
        "no_protein_blocks": 2,
        "no_cross_blocks": 2,
        "no_peptide_blocks": 2,
        "no_transition_layers": 1,
        "no_resnet_blocks": 2,
        "no_angles": 7,
        "trans_scale_factor": 10,
        "epsilon": 1e-12,
        "inf": 1e5,
        "blosum": False,
        "model_type": ModelType.REGRESSION,
        "peptide_maxlen": 16,
        "protein_maxlen": 200,
    },
)

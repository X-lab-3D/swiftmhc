from typing import Dict, Union
from copy import deepcopy as copy
import logging

from torch.nn import Embedding
from torch.nn.modules.transformer import TransformerEncoder
import torch
import ml_collections

from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.loss import compute_plddt
from openfold.data.data_transforms import make_atom14_masks
from openfold.data.data_pipeline import make_sequence_features
from openfold.utils.loss import find_structural_violations
from openfold.utils.feats import atom14_to_atom37
from openfold.model.primitives import LayerNorm

from .position_encoding import PositionalEncoding
from .cross_structure_module import CrossStructureModule
from ..domain.amino_acid import AMINO_ACID_DIMENSION
from ..models.data import TensorDict
from ..tools.amino_acid import one_hot_decode_sequence
from .transform import DebuggableTransformerEncoderLayer
from .ipa import DebuggableInvariantPointAttention as IPA


_log = logging.getLogger(__name__)


class Predictor(torch.nn.Module):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(Predictor, self).__init__()

        structure_module_config = copy(config.structure_module)
        structure_module_config.c_s = 32
        structure_module_config.c_z = 1
        structure_module_config.no_blocks = 2
        structure_module_config.no_heads_ipa = 2

        self.loop_maxlen = 16
        self.protein_maxlen = 40

        self.n_head = structure_module_config.no_heads_ipa

        self.pos_enc = PositionalEncoding(structure_module_config.c_s, self.loop_maxlen)

        self.loop_enc = TransformerEncoder(
            DebuggableTransformerEncoderLayer(structure_module_config.c_s,
                                              self.n_head,
                                              batch_first=True,
                                              norm_first=True),
            structure_module_config.no_blocks
        )

        self.n_ipa_repeat = structure_module_config.no_blocks

        self.protein_dist_norm = torch.nn.LayerNorm((self.protein_maxlen, self.protein_maxlen, 1))

        self.protein_ipa = IPA(structure_module_config.c_s,
                               structure_module_config.c_z,
                               structure_module_config.c_ipa,
                               structure_module_config.no_heads_ipa,
                               structure_module_config.no_qk_points,
                               structure_module_config.no_v_points)
        self.protein_ipa.inf = 1e22

        self.protein_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            LayerNorm(structure_module_config.c_s)
        )

        self.cross = CrossStructureModule(**structure_module_config)

        c_affinity = 128

        #self.aff_norm = LayerNorm(structure_module_config.c_s)

        #self.aff_trans = torch.nn.Sequential(
        #    torch.nn.Linear(structure_module_config.c_s, 10),
        #    torch.nn.GELU(),
        #    torch.nn.Linear(10, structure_module_config.c_s),
        #    torch.nn.Dropout(0.1),
        #    torch.nn.LayerNorm(structure_module_config.c_s)
        #)

        #mlp_input_size = self.loop_maxlen * structure_module_config.c_s * 2
        #mlp_input_size = self.loop_maxlen * self.protein_maxlen * structure_module_config.no_heads_ipa

        self.aff_mlp = torch.nn.Sequential(
            torch.nn.Linear(structure_module_config.c_s, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, 1),
        )

    def forward(self, batch: TensorDict) -> TensorDict:
        """
            Returns:
                single:               [batch_size, loop_len, c_s]
                aatype:               [batch_size, loop_len]  (int)
                residue_index:        [batch_size, loop_len]  (int)
                affinity:             [batch_size]

                frames:               [n_blocks, batch_size, loop_len, 4, 4]
                sidechain_frames:     [n_blocks, batch_size, loop_len, 8, 4, 4]
                unnormalized_angles:  [n_blocks, batch_size, loop_len, 7, 2]
                angles:               [n_blocks, batch_size, loop_len, 7, 2]
                positions:            [n_blocks, batch_size, loop_len, 14, 3]
                states:               [n_blocks, batch_size, loop_len, c_s]
        """

        # [batch_size, loop_len, c_s]
        loop_seq = batch["loop_sequence_onehot"]
        batch_size = loop_seq.shape[0]

        # positional encoding
        loop_pos_enc = self.pos_enc(loop_seq)

        # self-attention on the loop
        loop_embd = self.loop_enc(loop_pos_enc,
                                  src_key_padding_mask=torch.logical_not(batch["loop_self_mask"]))

        # store the attention weights, for debugging
        loop_enc_atts = []
        for layer in self.loop_enc.layers:
            loop_enc_atts.append(layer.last_att)
        # [n_layer, batch_size, n_head, loop_len, loop_len]
        loop_enc_atts = torch.stack(loop_enc_atts)

        # structure-based self-attention on the protein
        protein_T = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])

        # [batch_size, protein_len, c_s]
        protein_embd = batch["protein_sequence_onehot"]
        protein_norm_dist = self.protein_dist_norm(batch["protein_distances"])

        protein_atts = []
        for _ in range(self.n_ipa_repeat):
            protein_embd, protein_att = self.protein_ipa(protein_embd,
                                                         protein_norm_dist,
                                                         protein_T,
                                                         batch["protein_self_mask"].float())
            protein_atts.append(protein_att.clone().detach())

        # store the attention weights, for debugging
        # [n_layer, batch_size, n_head, protein_len, protein_len]
        protein_atts = torch.stack(protein_atts)

        protein_embd = self.protein_norm(protein_embd)

        # cross attention and loop structure prediction
        output = self.cross(batch["loop_aatype"],
                            loop_embd,
                            batch["loop_cross_mask"],
                            protein_embd,
                            batch["protein_cross_mask"],
                            protein_T)

        output["loop_self_attention"] = loop_enc_atts
        output["protein_self_attention"] = protein_atts

        # amino acid sequence: [1, 0, 2, ... ] meaning : Ala, Met, Cys
        # [batch_size, loop_len]
        output["aatype"] = batch["loop_aatype"]

        # amino acid sequence index: [0, 1, 2, 3, 4, ... ], representing the order of amino acids
        # [batch_size, loop_len]
        output["residue_index"] = torch.arange(0,
                                               loop_seq.shape[1], 1,
                                               dtype=torch.int64,
                                               device=loop_seq.device
        ).unsqueeze(dim=0).expand(batch_size, -1)

        # whether the heavy atoms exists or not
        # for each loop residue
        # [batch_size, loop_len, 14] (true or false)
        output = make_atom14_masks(output)

        # adding hydrogens:
        # [batch_size, loop_len, 37, 3]
        output["final_atom_positions"] = atom14_to_atom37(output["final_positions"], output)

        # [batch_size, n_heads, loop_len, protein_len]
        #cross_att = output["cross_attention"]

        # [batch_size, loop_maxlen, c_s]
        updated_s_loop = output["single"]

        # [batch_size, loop_maxlen]
        probabilities = self.aff_mlp(updated_s_loop).reshape(batch_size, -1)

        # [batch_size]
        output["affinity"] = torch.sum(probabilities, dim=1)

        return output



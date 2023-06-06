from typing import Optional
import logging

import torch

from torch.nn.modules.transformer import TransformerEncoderLayer


_log = logging.getLogger(__name__)


class DebuggableTransformerEncoderLayer(TransformerEncoderLayer):
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:

        x, a = self.self_attn(x, x, x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              average_attn_weights=False,
                              need_weights=True)

        # for debugging
        # [batch_size, n_head, x_len, x_len]
        self.last_att = a.clone().detach()

        return self.dropout(x)


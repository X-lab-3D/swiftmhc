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

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:

        x = src

        x = self.dropout(x)

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

import mlx.core as mx
from mlx import nn
from typing import Optional

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder
from mlx.nn.layers.distributed import shard_inplace, shard_linear


class QwenTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = QwenEncoder()

    def shard(self, group: Optional[mx.distributed.Group] = None):
        """
        Shard the 7B vision-language encoder across multiple devices.

        This is an optional optimization for very memory-constrained scenarios.
        The encoder has 28 transformer layers with 28 attention heads each.

        Args:
            group: The distributed group to shard across. If None, will initialize
                   a new group.
        """
        group = group or mx.distributed.init()
        N = group.size()

        # No sharding needed for single device
        if N == 1:
            return

        print(f"Sharding QwenTextEncoder (7B) across {N} devices...")

        # Shard each encoder layer
        for idx, layer in enumerate(self.encoder.layers):
            # Set sharding group reference for potential communication
            layer.sharding_group = group

            # Divide attention heads across devices
            if hasattr(layer, 'self_attn'):
                original_heads = layer.self_attn.num_attention_heads
                original_kv_heads = layer.self_attn.num_key_value_heads

                # Divide both query heads and key-value heads by N
                # This maintains the GQA ratio (28:4 = 7:1 becomes 7:1 per device with N=4)
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads //= N

                # Update num_key_value_groups after sharding
                layer.self_attn.num_key_value_groups = layer.self_attn.num_attention_heads // layer.self_attn.num_key_value_heads

                # CRITICAL: Also divide hidden_size since it's used in reshape operations
                # After sharding, the actual tensor dimension is hidden_size/N
                layer.self_attn.hidden_size //= N

                # Shard attention projections (all-to-sharded)
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )

                # Shard output projection (sharded-to-all)
                shard_inplace(layer.self_attn.o_proj, "sharded-to-all", group=group)

                if idx == 0:
                    print(f"  Layer 0: Divided {original_heads} query heads into {layer.self_attn.num_attention_heads} per device")
                    print(f"  Layer 0: Divided {original_kv_heads} key-value heads into {layer.self_attn.num_key_value_heads} per device (GQA ratio maintained)")

            # Shard MLP layers
            if hasattr(layer, 'mlp'):
                # Shard gate and up projections (all-to-sharded)
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
                # Shard down projection (sharded-to-all)
                shard_inplace(layer.mlp.down_proj, "sharded-to-all", group=group)

        print(f"[OK] Encoder sharding complete: {len(self.encoder.layers)} layers sharded")

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        hidden_states = self.encoder(input_ids, attention_mask)

        prompt_embeds, encoder_attention_mask = QwenTextEncoder._process_text_embeddings_mlx(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            drop_idx=34,
            dtype=mx.bfloat16,
        )

        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _process_text_embeddings_mlx(hidden_states, attention_mask, drop_idx=1, dtype=mx.float32):
        split_hidden_states = QwenTextEncoder._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [mx.ones(e.shape[0], dtype=mx.int32) for e in split_hidden_states]
        max_seq_len = max([e.shape[0] for e in split_hidden_states])

        padded_embeds = []
        for u in split_hidden_states:
            current_len = u.shape[0]
            hidden_dim = u.shape[1]
            if current_len < max_seq_len:
                padding = mx.zeros((max_seq_len - current_len, hidden_dim), dtype=u.dtype)
                padded = mx.concatenate([u, padding], axis=0)
            else:
                padded = u
            padded_embeds.append(padded)

        prompt_embeds = mx.stack(padded_embeds, axis=0)

        padded_masks = []
        for mask in attn_mask_list:
            current_len = mask.shape[0]
            if current_len < max_seq_len:
                padding = mx.zeros(max_seq_len - current_len, dtype=mask.dtype)
                padded = mx.concatenate([mask, padding], axis=0)
            else:
                padded = mask
            padded_masks.append(padded)

        encoder_attention_mask = mx.stack(padded_masks, axis=0)
        prompt_embeds = prompt_embeds.astype(dtype)
        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _extract_masked_hidden(hidden_states, attention_mask):
        batch_size = hidden_states.shape[0]
        split_hidden_states = []
        for i in range(batch_size):
            mask = attention_mask[i]
            valid_length = mx.sum(mask).item()
            valid_length = int(valid_length)
            valid_hidden = hidden_states[i, :valid_length, :]
            split_hidden_states.append(valid_hidden)
        return split_hidden_states

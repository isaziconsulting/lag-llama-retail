# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gluonts.torch.distributions import DistributionOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluonts.torch.util import lagged_sequence_values, unsqueeze_expand

from ...gluon_utils.scalers.robust_scaler import RobustScaler

@dataclass
class LTSMConfig:
    feature_size: int = 3 + 6  # target + loc + scale + time features
    block_size: int = 2048
    n_layer: int = 32
    n_head: int = 32
    n_embd_per_head: int = 128
    rope_scaling: Optional[dict] = None
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd_per_head * config.n_head)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd_per_head * config.n_head)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), use_kv_cache)
        y = x + self.mlp(self.rms_2(x))
        return y


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, device, dtype, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        # query projections for all heads, but in a batch
        self.q_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            config.n_embd_per_head * config.n_head,
            bias=False,
        )
        # key, value projections
        self.kv_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            2 * config.n_embd_per_head * config.n_head,
            bias=False,
        )
        # output projection
        self.c_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            config.n_embd_per_head * config.n_head,
            bias=False,
        )

        self.n_head = config.n_head
        self.n_embd_per_head = config.n_embd_per_head
        self.block_size = config.block_size
        self.dropout = config.dropout

        self.rope_scaling = config.rope_scaling
        self._rope_scaling_validation()

        self._init_rope()
        self.kv_cache = None

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.n_embd_per_head, max_position_embeddings=self.block_size
            )
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "nope":
                self.rotary_emb = None
            elif scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.n_embd_per_head,
                    max_position_embeddings=self.block_size,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.n_embd_per_head,
                    max_position_embeddings=self.block_size,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
            "linear",
            "dynamic",
            "nope",
        ]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_type in ["linear", "dynamic"]:
            if (
                rope_scaling_factor is None
                or not isinstance(rope_scaling_factor, float)
                or rope_scaling_factor < 1.0
            ):
                raise ValueError(
                    f"`rope_scaling`'s factor field must be an float >= 1, got {rope_scaling_factor}"
                )

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(x)
        k, v = self.kv_proj(x).split(self.n_embd_per_head * self.n_head, dim=2)

        cache_initialized = self.kv_cache is not None
        if use_kv_cache:
            # Optimized for single next prediction
            if cache_initialized:
                # Update cache
                k = torch.cat([self.kv_cache[0], k], dim=1)[:, 1:]
                v = torch.cat([self.kv_cache[1], v], dim=1)[:, 1:]
                self.kv_cache = k, v
            else:
                # Build cache
                self.kv_cache = k, v

        k = k.view(B, -1, self.n_head, self.n_embd_per_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, -1, self.n_head, self.n_embd_per_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, -1, self.n_head, self.n_embd_per_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        T_true = k.size(2)
        if self.rotary_emb is not None:
            if use_kv_cache and cache_initialized:
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=T_true)
                q, _ = apply_rotary_pos_emb(q, k, cos, sin, position_ids=[-1])
                
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=T_true)
                _, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)
            else:
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=T)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        # When using kv cache at inference, is_causal=False since decoder is causal, at each generation step we want
        # to avoid recalculating the same previous token attention

        if use_kv_cache and cache_initialized:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)

        return y


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd_per_head * config.n_head
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(
            config.n_embd_per_head * config.n_head, n_hidden, bias=False
        )
        self.c_fc2 = nn.Linear(
            config.n_embd_per_head * config.n_head, n_hidden, bias=False
        )
        self.c_proj = nn.Linear(
            n_hidden, config.n_embd_per_head * config.n_head, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        # keep RMSNorm in float32
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class LagLlamaModel(nn.Module):
    def __init__(
        self,
        context_length: int,
        max_context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        n_embd_per_head: int,
        n_head: int,
        lags_seq: List[int],
        distr_output: DistributionOutput,
        rope_scaling=None,
        num_parallel_samples: int = 100,
        time_feat: bool = True,
        num_feat_dynamic_real: int = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.lags_seq = lags_seq
        self.num_time_dims = 3 if time_feat else 0
        self.num_lag_dims = input_size * (len(self.lags_seq)) + 2 * input_size
        self.num_feat_dynamic_real = num_feat_dynamic_real

        feature_size = self.num_lag_dims + self.num_time_dims + num_feat_dynamic_real

        config = LTSMConfig(
            n_layer=n_layer,
            n_embd_per_head=n_embd_per_head,
            n_head=n_head,
            block_size=max_context_length,
            feature_size=feature_size,
            rope_scaling=rope_scaling,
            dropout=dropout,
        )
        self.num_parallel_samples = num_parallel_samples

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        elif scaling == "robust":
            self.scaler = RobustScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_args_proj(
            config.n_embd_per_head * config.n_head
        )

        # Input layers
        time_embed_capacity = 0.1 # Amount of embedding dimensions allocated to time features
        feat_dyn_embed_capacity = 0.1 # Amount of embedding dimensions allocated to other dynamic features

        num_embed_dims = config.n_embd_per_head * config.n_head
        num_time_embed_dims = math.floor(num_embed_dims * time_embed_capacity)
        num_feat_dyn_embed_dims = math.floor(num_embed_dims * feat_dyn_embed_capacity)
        # Rest of the embedding are given to the lagged targets
        num_lag_embed_dims = num_embed_dims - num_time_embed_dims - num_feat_dyn_embed_dims

        # self.embed_lags = nn.Linear(self.num_lag_dims, num_lag_embed_dims)
        # self.embed_time = nn.Linear(self.num_time_dims, num_time_embed_dims)
        # self.embed_dynamic_real = nn.Linear(num_feat_dynamic_real, num_feat_dyn_embed_dims)
        # self.interaction_layer = nn.Linear(num_embed_dims, num_embed_dims)


        self.transformer = nn.ModuleDict(
            dict(
                embed_lags = nn.Linear(self.num_lag_dims, num_lag_embed_dims),
                embed_time = nn.Linear(self.num_time_dims, num_time_embed_dims),
                embed_dynamic_real = nn.Linear(num_feat_dynamic_real, num_feat_dyn_embed_dims),
                interaction_layer = nn.Linear(num_embed_dims, num_embed_dims),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd_per_head * config.n_head),
            )
        )
        self.y_cache = False  # used at time of inference when kv cached is used

    def _init_weights(self, module: nn.Module) -> None:            
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def load_partial_weights(self, partial_weights_ckpt_path, device, freeze_transformer=False) -> None:
        checkpoint = torch.load(partial_weights_ckpt_path, device)
        # Remove the 'model.' prefix from the keys and filter only the transformer layers (excluding embedding layers) and distribution head layer
        valid_keys = ['transformer.h', 'transformer.ln_f']
        filtered_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if any(substring in k for substring in valid_keys)}
        self.load_state_dict(filtered_state_dict, strict=False)

        if freeze_transformer:
            # Freeze the transform layers that were loaded
            for name, param in self.named_parameters():
                if 'transformer.h' in name or 'transformer.ln_f' in name:
                    param.requires_grad = False

    def prepare_input(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        future_feat_dynamic_real: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        scaled_past_target, loc, scale = self.scaler(
            past_target, past_observed_values
        )  # Data is standardized (past_observed_values is passed as "weights" parameter) # (bsz, context_length+max(self.lags_seq)

        # In the below code, instead of max(self.lags_seq), it was previously -self.context_length
        if future_target is not None:
            input = torch.cat(
                (
                    scaled_past_target[..., max(self.lags_seq) :],  # Just the context
                    (future_target[..., :-1] - loc)
                    / scale,  # Not sure about the -1 here. Maybe so since the last value isn't used in the model for prediction of any new values. also if the prediction length is 1, this doesn't really affect anything
                ),
                dim=-1,
            )  # Shape is (bsz, context_length+(pred_len-1))
        else:
            input = scaled_past_target[..., max(self.lags_seq) :]
        if past_time_feat is not None:
            time_feat = (
                torch.cat(
                    (
                        past_time_feat[..., max(self.lags_seq) + 1:, :],
                        future_time_feat,
                    ),
                    dim=1,
                )
                if future_time_feat is not None
                else past_time_feat[..., max(self.lags_seq) + 1:, :]
            )

        if past_feat_dynamic_real is not None:
            feat_dynamic_real = (
                torch.cat(
                    (
                        past_feat_dynamic_real[..., max(self.lags_seq) + 1:, :],
                        future_feat_dynamic_real,
                    ),
                    dim=1,
                )
                if future_feat_dynamic_real is not None
                else past_feat_dynamic_real[..., max(self.lags_seq) + 1:, :]
            )

        prior_input = (
            past_target[..., : max(self.lags_seq)] - loc
        ) / scale  # This the history used to construct lags.  # bsz, max(self.lags_seq)

        lags = lagged_sequence_values(
            self.lags_seq, prior_input, input, dim=-1
        )  # Lags are added as an extra dim. Shape is (bsz, context_length+(pred_len-1), len(self.lags_seq))

        static_feat = torch.cat(
            (loc.abs().log1p(), scale.log()), dim=-1
        )  # (bsz, 2) (loc and scale are concatenated)
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=lags.shape[-2]
        )  # (bsz, context_length+(pred_len-1), 2)
        # expanded_static_feat: (bsz, context_length+(pred_len-1), len(self.lags_seq) + 2); (bsz, 1); (bsz, 1)

        feature_list = [lags, expanded_static_feat]
        if past_time_feat is not None:
            feature_list.append(time_feat)
        if past_feat_dynamic_real is not None:
            feature_list.append(feat_dynamic_real)

        return torch.cat(feature_list, dim=-1), loc, scale

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        future_feat_dynamic_real: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        # if past_time_feat is not None:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            future_feat_dynamic_real=future_feat_dynamic_real,
        )  # return: (bsz, context_length+(pred_len-1), len(self.lags_seq) + 2); (bsz, 1); (bsz, 1)
        # To use kv cache for inference and pass recent token to transformer
        if use_kv_cache and self.y_cache:
            # Only use the most recent one, rest is in cache
            transformer_input = transformer_input[:, -1:]

        # Embedding each type of input
        lags_emb = self.transformer.embed_lags(transformer_input[..., :self.num_lag_dims])
        time_emb = self.transformer.embed_time(transformer_input[..., self.num_lag_dims: self.num_lag_dims + self.num_time_dims])
        dynamic_emb = self.transformer.embed_dynamic_real(transformer_input[..., -self.num_feat_dynamic_real:])

        # Concatenate all embeddings
        x_concat = torch.cat([lags_emb, time_emb, dynamic_emb], dim=-1)

        # Apply interaction layer
        x = self.transformer.interaction_layer(x_concat)

        # # forward the LLaMA model itself
        # x = self.transformer.wte(
        #     transformer_input
        # )  # token embeddings of shape (b, t, n_embd_per_head*n_head) # (bsz, context_length+(pred_len-1), n_embd_per_head*n_head)

        for block in self.transformer.h:
            x = block(x, use_kv_cache)
        x = self.transformer.ln_f(
            x
        )  # (bsz, context_length+(pred_len-1), n_embd_per_head*n_head)
        if use_kv_cache:
            self.y_cache = True
        params = self.param_proj(
            x
        )  # (bsz, context_length+(pred_len-1)) ; (bsz, context_length+(pred_len-1))
        return params, loc, scale

    def reset_cache(self) -> None:
        """
        Resets all cached key-values in attention.
        Has to be called after prediction loop in predictor
        """
        self.y_cache = False 
        for block in self.transformer.h:
            block.attn.kv_cache = None

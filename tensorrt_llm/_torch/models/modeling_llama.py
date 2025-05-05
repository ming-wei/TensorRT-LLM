from typing import Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm._torch.models.context_logit_mode import ContextLogitMode
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class LlamaRotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        config: LlamaConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            config,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            device=device,
            rope_type="default" if config.rope_scaling is None else "llama3")


class LlamaAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        if model_config.fuse_pos_embd:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )
        else:
            pos_embd_params = None
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            rotary_emb=LlamaRotaryEmbedding(config),
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


class LlamaDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.self_attn = LlamaAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        ignore_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        print("ignore_context_logits is ", ignore_context_logits)
        skip_sdpa_for_context = not ignore_context_logits and all(x == 0 for x in attn_metadata.num_context_logits)

        assert hidden_states is not None
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            skip_sdpa_for_context=skip_sdpa_for_context,
            **kwargs,
        )

        assert hidden_states is not None

        skip_post_attention_layernorm_and_mlp = skip_sdpa_for_context and attn_metadata.num_generations == 0

        if not skip_post_attention_layernorm_and_mlp:
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        else:
            print("post_attention_layernorm and mlp skipped")
        return hidden_states, residual


class LlamaModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.padding_idx = config.pad_token_id

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None

        assert len(self.layers) >= 1, "LlamaModel must have at least 1 layer"
        assert attn_metadata.num_context_logits is not None, "num_context_logits must be provided"

        # TODO: implement context_logit_mode for layers
        ENABLE_SKIP_SDPA_FOR_CONTEXT = True
        layers_len = len(self.layers)
        # MINWEI
        for i in range(layers_len):
            decoder_layer = self.layers[i]
            if ENABLE_SKIP_SDPA_FOR_CONTEXT:
                ignore_context_logits = (i < layers_len - 1)
            else:
                ignore_context_logits = True
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual,
                                                    ignore_context_logits=ignore_context_logits
                                                    )

        skip_sdpa_for_context = ENABLE_SKIP_SDPA_FOR_CONTEXT and all(x == 0 for x in attn_metadata.num_context_logits)
        skip_norm = skip_sdpa_for_context and attn_metadata.num_generations == 0
        if not skip_norm:
            hidden_states, _ = self.norm(hidden_states, residual)

        assert hidden_states is not None
        return hidden_states


@register_auto_model("LlamaForCausalLM")
class LlamaForCausalLM(DecoderModelForCausalLM[LlamaModel, LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        super().__init__(LlamaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

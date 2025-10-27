#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:28:43 2024

@author: umbertocappellazzo
"""

import torch 
import torch.nn as nn
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer, LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache
from typing import Optional, Tuple
import math
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class LoRA_config:
    RANK: int
    ALPHA: int = 1
    IS_LLAMA3: bool = False
    IS_TINYLLAMA: bool = False
    
    

class LlamaSdpaAttention_lora(LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, lora_config: LoRA_config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        self.rank = lora_config.RANK
        self.scaling = lora_config.ALPHA/self.rank
        
        hid_size = config.hidden_size
        self.lora_down_Q = nn.Linear(hid_size, round(hid_size/self.rank), bias= False)
        self.lora_down_V = nn.Linear(hid_size, round(hid_size/self.rank), bias= False)
        self.lora_up_Q = nn.Linear(round(hid_size/self.rank), hid_size, bias= False)
        
        if lora_config.IS_LLAMA3: # grouped query attention (GQA) in action!! 
            self.lora_up_V = nn.Linear(round(hid_size/self.rank), hid_size//4, bias= False)
        elif lora_config.IS_TINYLLAMA:
            self.lora_up_V = nn.Linear(round(hid_size/self.rank), hid_size//8, bias= False)
        else:    
            self.lora_up_V = nn.Linear(round(hid_size/self.rank), hid_size, bias= False)
        
        # It is possible to apply LoRA to K and O matrices just in case.
        
        #self.lora_down_K = nn.Linear(hid_size, round(hid_size/self.rank), bias= False)
        #self.lora_up_K = nn.Linear(round(hid_size/self.rank), hid_size, bias= False)
        #self.lora_down_O = nn.Linear(hid_size, round(hid_size/self.rank), bias= False)
        #self.lora_up_O = nn.Linear(round(hid_size/self.rank), hid_size, bias= False)
    
        nn.init.zeros_(self.lora_down_Q.weight)
        nn.init.kaiming_uniform_(self.lora_up_Q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down_V.weight)
        nn.init.kaiming_uniform_(self.lora_up_V.weight, a=math.sqrt(5))
        #nn.init.zeros_(self.lora_down_K.weight)
        #nn.init.kaiming_uniform_(self.lora_up_K.weight, a=math.sqrt(5))
        #nn.init.zeros_(self.lora_down_O.weight)
        #nn.init.kaiming_uniform_(self.lora_up_O.weight, a=math.sqrt(5))
        
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    
    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        Q_lora = self.lora_up_Q(self.lora_down_Q(hidden_states))
        V_lora = self.lora_up_V(self.lora_down_V(hidden_states))
        #K_lora = self.lora_up_K(self.lora_down_K(hidden_states))
        
        query_states = query_states + Q_lora*self.scaling
        value_states = value_states + V_lora*self.scaling 
        #key_states = key_states + K_lora*self.scaling 

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output, attn = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if not output_attentions:
            attn = None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        
        #O_lora = self.lora_up_O(self.lora_down_O(hidden_states))
        
        attn_output = self.o_proj(attn_output) #+ O_lora*self.scaling

        return attn_output, attn, past_key_value

class LlamaForCausalLM_lora(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: LlamaConfig, lora_config: LoRA_config):
        super().__init__(config)
        self.lora_config= lora_config
        self.model = LlamaModel_lora(config, lora_config)
        
class LlamaModel_lora(LlamaModel):
    def __init__(self, config: LlamaConfig, lora_config: LoRA_config):
        super().__init__(config)
        self.lora_config= lora_config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_lora(config, layer_idx, lora_config) for layer_idx in range(config.num_hidden_layers)]
        )

class LlamaDecoderLayer_lora(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx, lora_config: LoRA_config):
        super().__init__(config, layer_idx)
        self.lora_config= lora_config
        
        self.self_attn = LlamaSdpaAttention_lora(config=config, layer_idx=layer_idx, lora_config=lora_config)

"""
Modified and Taken from: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight
"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from minigpt4.models.base_model import BaseModel
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
from minigpt4.models.eva_vit import create_eva_vit_g
from transformers import AutoTokenizer

from local import blip2_path, google_bert_path

class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        tokenizer = AutoTokenizer.from_pretrained(blip2_path)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        
        
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(blip2_path)
        encoder_config.encoder_width = vision_width
        
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, cross_layer_num, vit_model_path
    ):
        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        visual_encoder, cross_layers = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision,cross_layer_num, vit_model_path
        )
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, cross_layers, ln_vision

    def load_from_pretrained(self, url_or_filename):
        
        checkpoint = torch.load(google_bert_path, map_location="cpu")
        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    
    return self


class LayerNorm(nn.LayerNorm):
    

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
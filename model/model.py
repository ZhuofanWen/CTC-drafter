import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download


class DraftModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            draft_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(draft_model_path)
        with open(draft_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.layer = Model(config,load_emb=True,path=base_model_name_or_path,bias=bias)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.layer.diff_device = True
            if not low_memory:
                self.layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.layer.layer_device = device

        else:
            self.layer.diff_device = False
        self.layer.to(self.base_model.dtype).to(device)


    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            draft_model_path=None,
            use_safetensor_weight = False,
            **kwargs,
    ):
        base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        )

        if use_safetensor_weight==False:
            
            configpath=os.path.join(draft_model_path,"config.json")
            if not os.path.exists(configpath):
                configpath = hf_hub_download(draft_model_path, "config.json")
            model = cls(
                base_model,
                base_model_path,
                configpath
            )
            load_model_path=os.path.join(draft_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(draft_model_path, "pytorch_model.bin")
            layer_state_dict = torch.load(load_model_path,
                                            map_location=base_model.device)
            model.layer.load_state_dict(layer_state_dict, strict=True)
            
        else: # load models that is the type of .safetensors
            configpath=os.path.join(draft_model_path,"config.json")
            load_model_path=os.path.join(draft_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(draft_model_path, "model.safetensors")
            model = cls(
                base_model,
                base_model_path,
                configpath
            )
            from safetensors.torch import load_file
            layer_state_dict = load_file(load_model_path, device="cuda")
            model.layer.load_state_dict(layer_state_dict, strict=True)
                        
        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
        if init:
            token = torch.argmax(orig[:, -1])
            token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states

            logits = self.layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
            if output_orig:
                return logits, outputs, orig, hidden_states, token
            return logits, hidden_states, token
        else:
            if output_orig:
                return outputs, orig, hidden_states



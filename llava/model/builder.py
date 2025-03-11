#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
from types import SimpleNamespace

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
import torch.nn as nn
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.casper_attention.builder import AVG_AttentionModel as AttentionModel
from .casper_attention.builder import build_casper_attention

def extract_mm_projector_weight(non_lora_trainables):
    new_state_dict = {}
    for old_key, value in non_lora_trainables.items():
        # Map old keys to new keys
        if "mm_projector.0.weight" in old_key:
            new_state_dict["0.weight"] = value
        elif "mm_projector.0.bias" in old_key:
            new_state_dict["0.bias"] = value
        elif "mm_projector.2.weight" in old_key:
            new_state_dict["2.weight"] = value
        elif "mm_projector.2.bias" in old_key:
            new_state_dict["2.bias"] = value

    return new_state_dict

def load_pretrained_model(model_path, model_base, model_name, is_wsi_feature=False, load_8bit=False, load_4bit=False, additional_model_path=None, mm_projector_path=None, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            # model.load_state_dict(non_lora_trainables, strict=False)
            
            if mm_projector_path is not None:
                print("============================")
                print("Loading MM PROJECTOR weights")
                casper_projector = ModifiedMMProjector()
                print(casper_projector)
                print("\nOriginal mm_projector weights:")
                print(casper_projector.mm_projector[0].weight[0])
                mm_projector_state = torch.load(mm_projector_path, map_location='cpu')

                if "mm_projector" not in mm_projector_path:
                    mm_projector_state = extract_mm_projector_weight(mm_projector_state)

                casper_projector.mm_projector.load_state_dict(mm_projector_state)
                casper_projector.to(device=device, dtype=torch.float16)
                print("\nAfter mm_projector weights:")
                print(casper_projector.mm_projector[0].weight[0])
                model.model.mm_projector = casper_projector.mm_projector
        
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
            

        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
            
            
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                # Testing using the code of mm projector:
                non_lora_trainables = torch.load("./llava/model_weight/original_mm_projector/mm_projector.bin", map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)


    
    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        if is_wsi_feature:
            print("============================")
            print("Loading additional module...")
            
            casper_config = SimpleNamespace(**{
                "model": "token", # using learnable tokens
                "num_blocks": 2,
                "embed_dim": 1024,
                "num_heads": 8,
                "dropout": 0.1,
                "output_dim": 1024,
                "num_token": 400
            })
            casper_model = build_casper_attention(casper_config)

            print("\nOriginal AttentionModel weights:")
            print(casper_model.fc.weight)

            additional_module_states = torch.load(additional_model_path, map_location='cpu')
            casper_model.load_state_dict(additional_module_states)
            casper_model.to(device=device, dtype=torch.float16)
            model.set_casper_modules(casper_model)
            print("\After AttentionModel weights:")
            print(casper_model.fc.weight)
            image_processor = None

            model.to(device=device, dtype=torch.float16)
            

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    print("Model loaded successfully")
    return tokenizer, model, image_processor, context_len

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class ModifiedMMProjector(nn.Module):
    def __init__(self):
        super(ModifiedMMProjector, self).__init__()
        self.mm_projector = nn.Sequential(
            nn.Linear(1024, 4096),  # First Linear Layer: 1024 -> 4096
            nn.GELU(),             # GELU activation
            nn.Linear(4096, 4096)  # Second Linear Layer: 4096 -> 4096
        )
    def forward(self, x):
        return self.mm_projector(x)

def load_fixed_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")  # Remove "model." prefix
        new_state_dict[new_key] = v
    return new_state_dict

if __name__ == "__main__":
    model_base = None
    model_name = get_model_name_from_path(model_path)
    is_wsi_feature = True
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, is_wsi_feature, device_map="auto", device="cuda")
    print(model)
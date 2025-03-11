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


from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .casper_attention.builder import build_casper_attention

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import torch.nn.functional as F

testing_llm_only = False # This remove the image features from the model to test the LLM(LoRA) only

def adaptivate_pooling(images, target_tokens):
    batch_size, _, feature_dim = images.shape
    first_token = images[:, 0:1, :]  # Shape: (batch_size, 1, feature_dim)
    other_tokens = images[:, 1:, :]  # Shape: (batch_size, num_token - 1, feature_dim)
    pooled_images = F.adaptive_avg_pool1d(other_tokens.permute(0, 2, 1), target_tokens - 1)  # Shape: (batch_size, 1024, 199)
    pooled_images = pooled_images.permute(0, 2, 1)  # Shape: (batch_size, 199, 1024)
    images = torch.cat([first_token, pooled_images], dim=1)
    return images

def shrink_feature_dim(images, target_dim):
    batch_size, seq_len, input_dim = images.shape
    if input_dim % target_dim != 0:
        raise ValueError("Input dimension must be divisible by target dimension when using avg pooling.")
    reduction_factor = input_dim // target_dim
    images = images.view(batch_size, seq_len, target_dim, reduction_factor)
    output = images.mean(dim=-1)
    return output

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

class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            print("=================================================")
            print("Loading vision tower and mm_projector from config")
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

        self.wsi_encoder = False
        self.casper_model = None
        
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None): # Need to seperate vision tower and mm_projector
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("==========================")
            print("Checking self.mm_projector")
            print(self.mm_projector)

    def is_wsi_encoder(self):
        return self.wsi_encoder
    
    def initialize_casper_modules(self, model_args, fsdp=None):
        print("===========================================")
        print("Initializing casper module for WSI features")
        self.wsi_encoder = model_args.wsi_encoder

        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter



        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = 1024
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            if model_args.resume_from_bin:
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

                print("============================")
                print("Loading MM PROJECTOR weights")
                casper_projector = ModifiedMMProjector()
                print(casper_projector)
                print("\nOriginal mm_projector weights:")
                print(casper_projector.mm_projector[0].weight[0])
                mm_projector_path = os.path.join(model_args.resume_from_bin, 'non_lora_trainables.bin')
                mm_projector_state = torch.load(mm_projector_path, map_location='cpu')

                if "mm_projector" not in mm_projector_path:
                    mm_projector_state = extract_mm_projector_weight(mm_projector_state)

                casper_projector.mm_projector.load_state_dict(mm_projector_state)
                print("\nAfter mm_projector weights:")
                print(casper_projector.mm_projector[0].weight[0])
                self.mm_projector = casper_projector.mm_projector

            else:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        self.vision_tower = None
        if self.wsi_encoder:
            from types import SimpleNamespace
            casper_config = SimpleNamespace(**{
                "model": "token", # using learnable tokens
                "num_blocks": 2,
                "embed_dim": 1024,
                "num_heads": 8,
                "dropout": 0.1,
                "output_dim": 1024,
                "num_token": 400,
            })
            self.casper_model = build_casper_attention(casper_config)

    def set_casper_modules(self, model):
        self.wsi_encoder = True
        self.vision_tower = None
        self.casper_model = model

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def set_casper_modules(self, model):
        self.get_model().set_casper_modules(model)

    def encode_images(self, images, wsi_encoder=False):
        if wsi_encoder:
            images = images.float()
            if len(images.shape) == 2: 
                batch_size = images.size(0)
                patch_size = images.size(1)
                images = images.view(batch_size, patch_size, -1)

            elif len(images.shape) == 4: 
                batch_size = images.size(1)
                patch_size = images.size(2)
                images = images.view(batch_size, patch_size, -1)
            
            images = self.get_model().casper_model(images)
            
            if images.shape[-1] != 1024:
                pad_size = 1024 - images.size(2)
                images = F.pad(images, (0, pad_size), mode='constant', value=0)

            # Load and forward a feature of normal image into the mm projector to check if the model is working.
            image_features = self.get_model().mm_projector(images)

        else:
            print("THIS is normal image.")
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, ref_images=None
    ):
        vision_tower = self.get_vision_tower()
        if (vision_tower is None and not self.get_model().is_wsi_encoder()) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (vision_tower is not None or self.get_model().is_wsi_encoder()) and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            
            wsi_embeddings = False if concat_images.shape[1] == 3 else True # simply use this to differentiate between WSI and normal images
            image_features = self.encode_images(concat_images, wsi_embeddings)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
            if ref_images is not None:
                ref_image_features = self.encode_images(ref_images, wsi_embeddings)
        else:
            wsi_embeddings = False if images.shape[1] == 3 else True
            image_features = self.encode_images(images, wsi_embeddings)
            if ref_images is not None:
                ref_image_features = self.encode_images(ref_images, wsi_embeddings)

        if testing_llm_only:
            image_features = torch.zeros((1, 1, 4096), device="cuda", dtype=image_features.dtype) # for testing llm only

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids): # cur_input_ids is one input_id
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx] # they use image feature here, but how do they attach the image feature to the input_ids?
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0: # image token, should be 1
                if ref_images is not None and cur_image_idx == 1:
                    cur_image_idx -= 1 
                    cur_image_features = ref_image_features[cur_image_idx]
                else:
                    cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features) # they use image feature here, but how do they attach the image feature to the input_ids?
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0] # what is the purpose of this? weird 8
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer): # I think here
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

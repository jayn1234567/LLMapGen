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

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


def _infer_vision_tower_type(vision_tower):
    explicit_type = getattr(vision_tower, 'mm_vision_tower_type', None)
    if explicit_type:
        return explicit_type

    class_name = vision_tower.__class__.__name__.lower()
    tower_name = str(getattr(vision_tower, 'vision_tower_name', '')).lower()
    if 'dinov3' in class_name or 'dinov3' in tower_name:
        return 'dinov3'
    if 'dinov2' in class_name or 'dinov2' in tower_name:
        return 'dinov2'
    if 'mobileclip' in class_name or 'mobileclip' in tower_name:
        return 'mobileclip'
    if 'clip' in class_name or 'clip' in tower_name:
        return 'clip'
    return ''


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

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
            if getattr(vision_tower, 'vision_tower_name', None) != model_args.vision_tower:
                print(
                    f"Replacing vision tower {getattr(vision_tower, 'vision_tower_name', 'unknown')} "
                    f"with {model_args.vision_tower}."
                )
                vision_tower = build_vision_tower(model_args)
                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower[0] = vision_tower
                else:
                    self.vision_tower = vision_tower
            else:
                vision_tower.load_model()

        self.config.use_mm_proj = True
        vision_tower_type = _infer_vision_tower_type(vision_tower)
        if vision_tower_type:
            self.config.mm_vision_tower_type = vision_tower_type
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.input_image_size = (
            getattr(vision_tower, '_target_size', None)
            or getattr(vision_tower, 'input_image_size', None)
            or getattr(model_args, 'input_image_size', None)
        )
        deepstack_visual_indexes = getattr(vision_tower, 'deepstack_visual_indexes', None)
        if deepstack_visual_indexes is not None:
            deepstack_visual_indexes = [int(idx) for idx in deepstack_visual_indexes]
        self.config.deepstack_visual_indexes = deepstack_visual_indexes
        self.config.disable_deepstack = deepstack_visual_indexes is None
        self.config.dino_variant = getattr(model_args, 'dino_variant', None)

        mm_projector = getattr(self, 'mm_projector', None)
        projector_needs_rebuild = False
        if mm_projector is not None and hasattr(mm_projector, 'config'):
            projector_needs_rebuild = mm_projector.config.get("mm_projector_type") != self.config.mm_projector_type
        elif mm_projector is not None:
            first_linear = next((module for module in mm_projector.modules() if isinstance(module, nn.Linear)), None)
            projector_needs_rebuild = first_linear is None or first_linear.in_features != self.config.mm_hidden_size

        if mm_projector is None or projector_needs_rebuild:
            if projector_needs_rebuild:
                print(
                    f"Rebuilding mm_projector for vision hidden size {self.config.mm_hidden_size} "
                    f"with projector type {self.config.mm_projector_type}."
                )
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        if hasattr(self.get_vision_tower(), 'set_llm_hidden_size'):
            self.get_vision_tower().set_llm_hidden_size(self.config.hidden_size)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        output = self.get_model().get_vision_tower()(images)
        if isinstance(output, (tuple, list)) and len(output) == 2:
            main_features, deepstack_features = output
        else:
            main_features, deepstack_features = output, None
        main_features = self.get_model().mm_projector(main_features.to(self.dtype))
        return main_features, deepstack_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        all_deepstack_features = None
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            encoded = self.encode_images(concat_images)
            if isinstance(encoded, tuple) and len(encoded) == 2:
                image_features, all_deepstack_features = encoded
            else:
                image_features = encoded
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            if hasattr(self.get_vision_tower(), 's2_image_size'):
                                img_size = self.get_vision_tower().s2_image_size
                            elif isinstance(self.get_vision_tower().config, dict):
                                img_size = self.get_vision_tower().config["image_cfg"]["image_size"]
                            else:
                                img_size = self.get_vision_tower().config.image_size

                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, img_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
            if isinstance(image_features, tuple) and len(image_features) == 2:
                image_features, all_deepstack_features = image_features

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_visual_pos_masks = []
        new_deepstack_token_features = None
        if all_deepstack_features is not None:
            new_deepstack_token_features = [[] for _ in range(len(all_deepstack_features))]
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_visual_pos_masks.append(torch.zeros(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device))
                if new_deepstack_token_features is not None:
                    for layer_idx in range(len(all_deepstack_features)):
                        hidden_size = all_deepstack_features[layer_idx].shape[-1]
                        new_deepstack_token_features[layer_idx].append(
                            torch.zeros(
                                cur_input_embeds.shape[0],
                                hidden_size,
                                dtype=all_deepstack_features[layer_idx].dtype,
                                device=cur_input_embeds.device,
                            )
                        )
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_visual_pos_mask = []
            cur_deepstack_token_features = None
            if new_deepstack_token_features is not None:
                cur_deepstack_token_features = [[] for _ in range(len(all_deepstack_features))]

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_visual_pos_mask.append(torch.zeros(cur_input_embeds_no_im[i].shape[0], dtype=torch.bool, device=cur_input_embeds_no_im[i].device))
                if cur_deepstack_token_features is not None:
                    for layer_idx in range(len(all_deepstack_features)):
                        hidden_size = all_deepstack_features[layer_idx].shape[-1]
                        cur_deepstack_token_features[layer_idx].append(
                            torch.zeros(
                                cur_input_embeds_no_im[i].shape[0],
                                hidden_size,
                                dtype=all_deepstack_features[layer_idx].dtype,
                                device=cur_input_embeds_no_im[i].device,
                            )
                        )
                if i < num_images:
                    image_feature_idx = cur_image_idx
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_visual_pos_mask.append(torch.ones(cur_image_features.shape[0], dtype=torch.bool, device=cur_image_features.device))
                    if cur_deepstack_token_features is not None:
                        for layer_idx, layer_features in enumerate(all_deepstack_features):
                            cur_deepstack_token_features[layer_idx].append(layer_features[image_feature_idx])

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_visual_pos_mask = [x.to(self.device) for x in cur_visual_pos_mask]
            if cur_deepstack_token_features is not None:
                cur_deepstack_token_features = [
                    [x.to(self.device) for x in layer_parts]
                    for layer_parts in cur_deepstack_token_features
                ]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_visual_pos_mask = torch.cat(cur_visual_pos_mask)
            if cur_deepstack_token_features is not None:
                cur_deepstack_token_features = [
                    torch.cat(layer_parts, dim=0)
                    for layer_parts in cur_deepstack_token_features
                ]

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_visual_pos_masks.append(cur_visual_pos_mask)
            if new_deepstack_token_features is not None:
                for layer_idx, layer_features in enumerate(cur_deepstack_token_features):
                    new_deepstack_token_features[layer_idx].append(layer_features)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_visual_pos_masks = [x[:tokenizer_model_max_length] for x in new_visual_pos_masks]
            if new_deepstack_token_features is not None:
                new_deepstack_token_features = [
                    [x[:tokenizer_model_max_length] for x in layer_features]
                    for layer_features in new_deepstack_token_features
                ]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        deepstack_features_padded = None
        if new_deepstack_token_features is not None:
            deepstack_features_padded = [[] for _ in range(len(new_deepstack_token_features))]
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        visual_pos_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_input_embeds[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_visual_mask) in enumerate(zip(new_input_embeds, new_labels, new_visual_pos_masks)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    visual_pos_mask[i, -cur_len:] = cur_visual_mask
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                if deepstack_features_padded is not None:
                    for layer_idx, layer_features in enumerate(new_deepstack_token_features):
                        cur_layer_features = layer_features[i]
                        deepstack_features_padded[layer_idx].append(torch.cat((
                            torch.zeros((max_len - cur_len, cur_layer_features.shape[1]), dtype=cur_layer_features.dtype, device=cur_layer_features.device),
                            cur_layer_features
                        ), dim=0))
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    visual_pos_mask[i, :cur_len] = cur_visual_mask
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                if deepstack_features_padded is not None:
                    for layer_idx, layer_features in enumerate(new_deepstack_token_features):
                        cur_layer_features = layer_features[i]
                        deepstack_features_padded[layer_idx].append(torch.cat((
                            cur_layer_features,
                            torch.zeros((max_len - cur_len, cur_layer_features.shape[1]), dtype=cur_layer_features.dtype, device=cur_layer_features.device)
                        ), dim=0))

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if deepstack_features_padded is not None:
            all_deepstack_features = [
                torch.stack(layer_features, dim=0)[visual_pos_mask]
                for layer_features in deepstack_features_padded
            ]

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if all_deepstack_features is None:
            visual_pos_mask = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, visual_pos_mask, all_deepstack_features

    def initialize_vision_tokenizer(self, model_args, tokenizer):
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

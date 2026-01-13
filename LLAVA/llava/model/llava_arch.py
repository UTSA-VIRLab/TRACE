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
# try to import the ImageModel class from the LLAVA.biovil_t.model module
try: #eval
    from LLAVA.biovil_t.model import ImageModel
    from LLAVA.biovil_t.pretrained import _download_biovil_t_image_model_weights
    from LLAVA.biovil_t.types import ImageEncoderType
    from LLAVA.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    from LLAVA.llava.model.multimodal_encoder.builder import build_vision_tower
    from LLAVA.llava.model.multimodal_projector.builder import build_vision_projector, build_image_pooler
    from LLAVA.llava.model.temporal_grounding import TemporalGroundingHead
except ImportError: #train
    from biovil_t.model import ImageModel
    from biovil_t.pretrained import _download_biovil_t_image_model_weights
    from biovil_t.types import ImageEncoderType
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    from llava.model.multimodal_encoder.builder import build_vision_tower
    from llava.model.multimodal_projector.builder import build_vision_projector, build_image_pooler
    from llava.model.temporal_grounding import TemporalGroundingHead
from .temporal_grounding_v3 import TemporalGroundingHeadV3, build_temporal_grounding_head_v3
from .temporal_grounding_v4 import TemporalGroundingHeadV4
from .temporal_grounding_v5 import TemporalGroundingHeadV5


class LlavaMetaModel:

    def __init__(self, config, mv_type='none'):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            self.image_pooler = build_image_pooler(config) if "pool" in mv_type else None
            
            # TemporalGroundingHead for temporal comparison mode
            if mv_type == "temporal":
                self.temporal_grounding_head = TemporalGroundingHeadV5(
                    hidden_dim=512,  # BioViL-T feature size
                    num_heads=8,
                    num_queries=20,
                    num_change_classes=3,
                    dropout=0.1
                )
            else:
                self.temporal_grounding_head = None

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_image_pooler(self):
        return self.image_pooler

    def get_temporal_grounding_head(self):
        return getattr(self, 'temporal_grounding_head', None)

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.mv_type = getattr(model_args, 'mv_type', False)

        if self.get_vision_tower() is None:
            if self.config.mm_vision_tower == 'biovil':
                biovilt_checkpoint_path = _download_biovil_t_image_model_weights()
                model_type = ImageEncoderType.RESNET50_MULTI_IMAGE
                vision_tower = ImageModel(img_encoder_type=model_type,
                                          joint_feature_size=128,
                                          pretrained_model_path=biovilt_checkpoint_path)
                # freeze vision_tower layers
                for p in vision_tower.parameters():
                    p.requires_grad = False
            else:
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
        self.config.mm_hidden_size = vision_tower.hidden_size if self.config.mm_vision_tower != 'biovil' else vision_tower.feature_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector',
                   None) is None or model_args.vision_tower == 'biovil':  # for biovil wrong weights are loaded from model shards, so we need to overwrite the vision projector again
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # unfreeze image pooler
        if getattr(self, "image_pooler", None) is not None:
            for p in self.image_pooler.parameters():
                p.requires_grad = True

        # Initialize TemporalGroundingHead for temporal mode
        mv_type = getattr(model_args, 'mv_type', 'none')
        if mv_type == "temporal":
            if getattr(self, 'temporal_grounding_head', None) is None:
                self.temporal_grounding_head = TemporalGroundingHeadV5(
                    hidden_dim=512,
                    num_heads=8,
                    num_queries=20,
                    num_change_classes=3,
                    dropout=0.1
                )
            # Make sure temporal head is trainable
            for p in self.temporal_grounding_head.parameters():
                p.requires_grad = True
            num_params = sum(p.numel() for p in self.temporal_grounding_head.parameters())
            print(f"Initialized TemporalGroundingHead with {num_params/1e6:.1f}M parameters")

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        if self.get_model().config.mm_vision_tower == 'biovil':
            image_features = image_features.patch_embeddings
            # flatten
            image_features = image_features.flatten(2).transpose(1, 2)

        image_features = self.get_model().mm_projector(image_features)

        return image_features

    def encode_images_raw(self, images):
        """Encode images WITHOUT projection - returns raw vision features for temporal grounding"""
        image_features = self.get_model().get_vision_tower()(images)
        if self.get_model().config.mm_vision_tower == 'biovil':
            image_features = image_features.patch_embeddings
            image_features = image_features.flatten(2).transpose(1, 2)
        return image_features  # [B, N, 512] raw BioViL-T features

    def encode_images_temporal(self, current_images, prior_images, gt_change=None):
        """Encode current and prior images for temporal comparison.
        
        Simple V1 approach: concatenate [prior, current] features.
        """
        temporal_head = self.get_temporal_grounding_head()
        
        # Get features for both images
        curr_features = self.encode_images(current_images)
        
        if prior_images is not None:
            prev_features = self.encode_images(prior_images)
            # V1: Simple concatenation
            projected_features = torch.cat([prev_features, curr_features], dim=1)
        else:
            projected_features = curr_features
        
        # If temporal head exists, also compute grounding outputs for aux loss
        grounding_outputs = None
        if temporal_head is not None and prior_images is not None:
            current_raw = self.encode_images_raw(current_images)
            prior_raw = self.encode_images_raw(prior_images)
            grounding_outputs = temporal_head(current_raw, prior_raw, gt_change=gt_change, return_attention=False)
        
        return projected_features, grounding_outputs


    def get_temporal_grounding_head(self):
        return self.get_model().get_temporal_grounding_head()

    def pad_embeddings(self, embeddings, num_imgs_present=None, num_imgs_past=None, padding_value=0):
        """
        Pad the embeddings to have the same number in each batch.

        Args:
        - embeddings (List[Tensor]): List of embedding tensors, each with shape (num_images, embedding_dim).
        - padding_value (float): Value to use for padding.

        Returns:
        - Tensor: Padded embeddings with shape (batch_size, max_num_images, embedding_dim).
        - Tensor: Mask indicating real data (1) and padding (0).
        """
        batch_size = len(embeddings)
        img_len = embeddings[0].shape[1]
        embedding_dim = embeddings[0].shape[2]
        max_num_images = max(emb.shape[0] for emb in embeddings)

        # Initialize padded embeddings and mask
        padded_embeddings = torch.full((batch_size, max_num_images, img_len, embedding_dim), padding_value, dtype=embeddings[0].dtype,
                                       device=embeddings[0].device)
        mask = torch.zeros(batch_size, max_num_images * img_len, dtype=torch.bool, device=embeddings[0].device)
        # create token type ids with 0 for present 1 for past, 2 for padding, of shape (batch_size, max_num_images * img_len)
        token_type_ids = torch.zeros(batch_size, max_num_images * img_len, dtype=torch.long, device=embeddings[0].device)
        if num_imgs_present is not None:
            # set token type ids for present to 1, for past to 2, 0 is padded elements
            for idx, (present_len, past_len) in enumerate(zip(num_imgs_present, num_imgs_past)):
                token_type_ids[idx, :present_len * img_len] = 1
                token_type_ids[idx, present_len * img_len:(present_len + past_len) * img_len] = 2

        # Pad each item in the batch
        for idx, emb in enumerate(embeddings):
            num_images = emb.shape[0]
            padded_embeddings[idx, :num_images] = emb
            mask[idx, :num_images * img_len] = 1

        return padded_embeddings.flatten(1, 2), mask, token_type_ids

    def pad_embeddings_mv(self, embeddings, padding_value=0):
        """
        Pad the embeddings to have the same number in each batch.

        Args:
        - embeddings (List[Tensor]): List of embedding tensors, each with shape (num_images, embedding_dim).
        - padding_value (float): Value to use for padding.

        Returns:
        - Tensor: Padded embeddings with shape (batch_size, max_num_images, embedding_dim).
        - Tensor: Mask indicating real data (1) and padding (0).
        """
        batch_size = len(embeddings)
        img_len = embeddings[0].shape[1]
        embedding_dim = embeddings[0].shape[2]
        max_num_images = max(emb.shape[0] for emb in embeddings)

        # Initialize padded embeddings and mask
        padded_embeddings = torch.full((batch_size, max_num_images, img_len, embedding_dim), padding_value, dtype=embeddings[0].dtype,
                                       device=embeddings[0].device)
        mask = torch.zeros(batch_size, max_num_images * img_len, dtype=torch.bool, device=embeddings[0].device)

        # Pad each item in the batch
        for idx, emb in enumerate(embeddings):
            num_images = emb.shape[0]
            padded_embeddings[idx, :num_images] = emb
            mask[idx, :num_images * img_len] = 1

        return padded_embeddings.flatten(1, 2), mask

    def encode_images_pooled(self, images, split_sizes, num_imgs_present, num_imgs_past, mv_type="pool_all"):
        image_pooler = self.get_image_pooler()
        image_features = self.get_model().get_vision_tower()(images)
        if self.get_model().config.mm_vision_tower == 'biovil':
            image_features = image_features.patch_embeddings
            # flatten
            image_features = image_features.flatten(2).transpose(1, 2)
        if split_sizes is not None:
            image_features = torch.split(image_features, split_sizes, dim=0)

            if mv_type == "pool_all":
                # merge present and past per batch
                present_features = [image_features[i] for i in range(len(num_imgs_present))]
                past_features = []
                i = 0
                for num_imgs_elem in num_imgs_past:
                    if num_imgs_elem != 0:
                        past_features.append(image_features[i + len(num_imgs_present)])
                        i += 1
                    else:
                        past_features.append(None)

                all_img_features = []
                for idx, (batch_num_present, batch_num_past) in enumerate(zip(num_imgs_present, num_imgs_past)):
                    if batch_num_past == 0:
                        all_img_features.append(present_features[idx])
                    else:
                        all_img_features.append(torch.cat((present_features[idx], past_features[idx]), dim=0))

                all_img_features, mask, token_type_ids = self.pad_embeddings(all_img_features, num_imgs_present, num_imgs_past)
                all_img_features = image_pooler(all_img_features, mask, token_type_ids)

            elif mv_type == "pool_concat":
                present_features = [image_features[i] for i in range(len(num_imgs_present))]
                past_features = [image_features[i + len(num_imgs_present)] for i in range(len(image_features) - len(num_imgs_present))]
                present_features, mask_present, _ = self.pad_embeddings(present_features)
                past_features, mask_past, _ = self.pad_embeddings(past_features)
                present_features = image_pooler(present_features, mask_present)
                past_features = image_pooler(past_features, mask_past)
                # TODO maybe max pool on past features to save tokens
                # concat present and past per batch if past is not empty
                all_img_features = []
                idx_present = 0
                idx_past = 0
                for batch_num_present, batch_num_past in zip(num_imgs_present, num_imgs_past):
                    if batch_num_past == 0:
                        all_img_features.append(present_features[idx_present])
                        idx_present += 1
                    else:
                        all_img_features.append(torch.cat((present_features[idx_present], past_features[idx_past]), dim=0))
                        idx_present += 1
                        idx_past += 1
        else:
            raise NotImplementedError
        if type(all_img_features) is list:
            split_sizes = [image.shape[0] for image in all_img_features]
            all_img_features = self.get_model().mm_projector(torch.cat(all_img_features, dim=0))
            all_img_features = torch.split(all_img_features, split_sizes, dim=0)

        else:
            all_img_features = self.get_model().mm_projector(all_img_features)
        return all_img_features

    def encode_images_pooled_mv(self, images, split_sizes):
        image_pooler = self.get_image_pooler()
        image_features = self.get_model().get_vision_tower()(images)
        if split_sizes is not None:
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features, mask = self.pad_embeddings_mv(image_features)
            image_features = image_pooler(image_features, mask)
        else:
            mask = torch.ones((image_features.shape[0], image_features.shape[1]), dtype=torch.bool, device=image_features[0].device)
            image_features = image_pooler(image_features, mask)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def get_image_pooler(self):
        return self.get_model().get_image_pooler()

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, prev_images=None, gt_change=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            # Temporal mode with cross-attention
            if getattr(self.config, 'mv_type') == "temporal" and prev_images is not None:
                all_features = []
                all_grounding = []
                
                for i, (curr_img, prev_img) in enumerate(zip(images, prev_images)):
                    if prev_img is not None:
                        # Use temporal cross-attention
                        curr_input = curr_img.unsqueeze(0) if curr_img.dim() == 3 else curr_img
                        prev_input = prev_img.unsqueeze(0) if prev_img.dim() == 3 else prev_img
                        features, grounding = self.encode_images_temporal(curr_input, prev_input, gt_change=gt_change)
                        all_features.append(features.flatten(0, 1))
                        all_grounding.append(grounding)
                    else:
                        # No prior - just encode current
                        curr_input = curr_img.unsqueeze(0) if curr_img.dim() == 3 else curr_img
                        features = self.encode_images(curr_input)
                        all_features.append(features.flatten(0, 1))
                        all_grounding.append(None)
                
                image_features = [f.to(self.device) for f in all_features]
                # Store grounding outputs for loss computation
                if any(g is not None for g in all_grounding):
                    valid_grounding = [g for g in all_grounding if g is not None]
                    self._temporal_grounding_outputs = {
                        'boxes': torch.stack([g['boxes'].squeeze(0) for g in valid_grounding]),
                        'box_classes': torch.stack([g['box_classes'].squeeze(0) for g in valid_grounding]),
                        'box_confidence': torch.stack([g['box_confidence'].squeeze(0) for g in valid_grounding]),
                        'patch_changes': torch.stack([g['patch_changes'].squeeze(0) for g in valid_grounding]),
                    }
            elif getattr(self.config, 'mv_type') == "concat":
                # Include prev_images if available (for temporal comparison)
                if prev_images is not None and any(p is not None for p in prev_images):
                    # Concatenate current and prior images
                    all_images = []
                    for img, prev_img in zip(images, prev_images):
                        all_images.append(img)
                        if prev_img is not None:
                            all_images.append(prev_img.unsqueeze(0) if prev_img.dim() == 3 else prev_img)
                    concat_images = torch.cat([image for image in all_images], dim=0)
                    image_features = self.encode_images(concat_images)
                    # Split back: alternating current, prev for each batch item
                    split_sizes = []
                    for img, prev_img in zip(images, prev_images):
                        split_sizes.append(img.shape[0])
                        if prev_img is not None:
                            split_sizes.append(prev_img.shape[0] if prev_img.dim() == 4 else 1)
                    image_features = torch.split(image_features, split_sizes, dim=0)
                    # Combine current and prev features for each batch item
                    combined_features = []
                    idx = 0
                    for i, prev_img in enumerate(prev_images):
                        curr_feat = image_features[idx].flatten(0, 1)
                        idx += 1
                        if prev_img is not None:
                            prev_feat = image_features[idx].flatten(0, 1)
                            idx += 1
                            # Concatenate along sequence dimension (prior first, then current)
                            combined_features.append(torch.cat([prev_feat, curr_feat], dim=0).to(self.device))
                        else:
                            combined_features.append(curr_feat.to(self.device))
                    image_features = combined_features
                else:
                    concat_images = torch.cat([image for image in images], dim=0)
                    image_features = self.encode_images(concat_images)
                    split_sizes = [image.shape[0] for image in images]
                    image_features = torch.split(image_features, split_sizes, dim=0)
                    image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
            if getattr(self.config, 'mv_type') == "pool_all":
                concat_images = torch.cat((torch.cat([image for image in images], dim=0),
                                           torch.cat([image for image in prev_images if image is not None],
                                                     dim=0)))  # first present, then past, all will be merged
                split_sizes = [image.shape[0] for image in images] + [image.shape[0] for image in prev_images if image is not None]
                num_imgs_present = [image.shape[0] if image is not None else 0 for image in images]
                num_imgs_past = [image.shape[0] if image is not None else 0 for image in prev_images]
                image_features = self.encode_images_pooled(concat_images, split_sizes, num_imgs_present, num_imgs_past, "pool_all")
            if getattr(self.config, 'mv_type') == "pool_concat":  # TODO make sure to allow empty past -> shorter sequence
                concat_images = torch.cat((torch.cat([image for image in images], dim=0),
                                           torch.cat([image for image in prev_images if image is not None],
                                                     dim=0)))  # first present, then past, all will be merged
                split_sizes = [image.shape[0] for image in images] + [image.shape[0] for image in prev_images if image is not None]
                num_imgs_present = [image.shape[0] if image is not None else 0 for image in images]
                num_imgs_past = [image.shape[0] if image is not None else 0 for image in prev_images]
                image_features = self.encode_images_pooled(concat_images, split_sizes, num_imgs_present, num_imgs_past, "pool_concat")
            if getattr(self.config, 'mv_type') == "pool":  # no past images
                concat_images = torch.cat([image for image in images], dim=0)
                split_sizes = [image.shape[0] for image in images]
                image_features = self.encode_images_pooled_mv(concat_images, split_sizes)
        else:
            if hasattr(self.config, 'mv_type') and getattr(self.config, 'mv_type') == "pool_all":
                image_features = self.encode_images_pooled(images, None).to(self.device)
            elif hasattr(self.config, 'mv_type') and getattr(self.config, 'mv_type') == "pool":
                image_features = self.encode_images_pooled_mv(images, None).to(self.device)
            else:
                # Handle prev_images for temporal mode (single image case)
                if prev_images is not None and getattr(self.config, 'mv_type', None) == "temporal":
                    image_features, grounding = self.encode_images_temporal(images, prev_images, gt_change=gt_change)
                    image_features = image_features.to(self.device)
                    if grounding is not None:
                        self._temporal_grounding_outputs = grounding
                # Handle prev_images for concat mode (single image case)
                elif prev_images is not None and getattr(self.config, 'mv_type', None) == "concat":
                    # Encode both current and prior images
                    curr_features = self.encode_images(images).to(self.device)
                    prev_features = self.encode_images(prev_images).to(self.device)
                    # Concatenate: prior first, then current (matches training)
                    image_features = torch.cat([prev_features, curr_features], dim=1)
                else:
                    image_features = self.encode_images(images).to(self.device)

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
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)  # TODO throws GPU error
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            max_len_orig = max(x.shape[0] for x in new_input_embeds)
            if max_len_orig > tokenizer_model_max_length:
                print(f"Truncating sequences of len {max_len_orig} to {tokenizer_model_max_length} to fit the model's input length")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

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

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

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
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

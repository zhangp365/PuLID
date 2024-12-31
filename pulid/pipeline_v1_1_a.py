import gc
import sys
sys.path.append('.')
import time
import cv2
import insightface
import numpy as np
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from pulid.encoders_transformer import IDFormer
from pulid.utils import is_torch2_available, sample_dpmpp_2m, sample_dpmpp_sde
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torchvision import transforms
to_tensor = transforms.ToTensor()

import attention_processor as attention

if is_torch2_available():
    from pulid.attention_processor import AttnProcessor2_0 as AttnProcessor
    from pulid.attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
else:
    from pulid.attention_processor import AttnProcessor, IDAttnProcessor


class PuLIDPipeline:
    def __init__(self, sdxl_repo='Lykon/dreamshaper-xl-lightning', device='cuda', *args, **kwargs):
        super().__init__()
        self.device = device

        # load base model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_repo, torch_dtype=torch.bfloat16, variant="fp16").to(device)
        self.pipe.watermark = None
        self.hack_unet_attn_layers(self.pipe.unet)

        # ID adapters
        self.id_adapter = IDFormer()

        # Initialize preprocessors
        self._init_preprocessors()

        gc.collect()
        self.load_pretrain()
        self.debug_img_list = []

    def _init_preprocessors(self):
        """初始化所有预处理器组件"""
        # 1. 初始化face helper
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device="cpu",
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device="cpu")

        # 2. 初始化CLIP视觉模型
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.clip_vision_model = model.to("cpu")
        
        # 设置转换参数
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        self.eva_transform_mean = (eva_transform_mean,) * 3 if not isinstance(eva_transform_mean, (list, tuple)) else eva_transform_mean
        self.eva_transform_std = (eva_transform_std,) * 3 if not isinstance(eva_transform_std, (list, tuple)) else eva_transform_std

        # 3. 初始化antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        self.app = FaceAnalysis(
            name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=0)

    def hack_unet_attn_layers(self, unet):
        id_adapter_attn_procs = {}
        index = 0
        for name, processor in unet.attn_processors.items():
            # print("name", name,"processor", type(processor))
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is not None:
               
                id_adapter_attn_procs[name] = IDAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    index_of_whole_embedding= index,
                ).to(unet.device, dtype=torch.bfloat16)
                if hidden_size == 1280:
                    index += 4
                else:
                    index += 2
            else:
                id_adapter_attn_procs[name] = AttnProcessor()
        print("index:", index)
        unet.set_attn_processor(id_adapter_attn_procs)
        self.id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values())

    def load_pretrain(self):
        hf_hub_download('guozinan/PuLID', 'pulid_v1.1.safetensors', local_dir='models')
        ckpt_path = 'models/pulid_v1.1.safetensors'
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    def get_id_embedding(self, image_list,num_zero):
        """
        Args:
            image in image_list: numpy rgb image, range [0, 255]
        """
        id_cond_list = []
        id_vit_hidden_list = []
        for ii, image in enumerate(image_list):
            self.face_helper.clean_all()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # get antelopev2 embedding
            face_info = self.app.get(image_bgr)
            if len(face_info) > 0:
                face_info = sorted(
                    face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
                )[
                    -1
                ]  # only use the maximum face
                id_ante_embedding = face_info['embedding']
                self.debug_img_list.append(
                    image[
                        int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                        int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                    ]
                )
            else:
                id_ante_embedding = None

            # using facexlib to detect and align face
            self.face_helper.read_image(image_bgr)
            self.face_helper.get_face_landmarks_5(only_center_face=True)
            self.face_helper.align_warp_face()
            if len(self.face_helper.cropped_faces) == 0:
                raise RuntimeError('facexlib align face fail')
            align_face = self.face_helper.cropped_faces[0]
            # incase insightface didn't detect face
            if id_ante_embedding is None:
                print('fail to detect face using insightface, extract embedding on align face')
                id_ante_embedding = self.handler_ante.get_feat(align_face)

            id_ante_embedding = torch.from_numpy(id_ante_embedding)
            if id_ante_embedding.ndim == 1:
                id_ante_embedding = id_ante_embedding.unsqueeze(0)

            # parsing
            input = to_tensor(cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB)).unsqueeze(0)
            parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[
                0
            ]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input)
            # only keep the face features
            face_features_image = torch.where(bg, white_image, self.to_gray(input))
            self.debug_img_list.append((face_features_image[0].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0))

            face_features_image = resize(
                face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC
            )
            face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
            id_cond_vit, id_vit_hidden = self.clip_vision_model(
                face_features_image, return_all_features=False, return_hidden=True, shuffle=False
            )
            id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
            id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

            id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

            id_cond_list.append(id_cond)
            id_vit_hidden_list.append(id_vit_hidden)

        id_uncond = torch.zeros_like(id_cond_list[0])
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden_list[0])):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden_list[0][layer_idx]))

        id_cond = torch.stack(id_cond_list, dim=1)
        id_vit_hidden = id_vit_hidden_list[0]
        for i in range(1, len(image_list)):
            for j, x in enumerate(id_vit_hidden_list[i]):
                id_vit_hidden[j] = torch.cat([id_vit_hidden[j], x], dim=1)
        id_embedding = self.id_adapter(id_cond, id_vit_hidden)
        uncond_id_embedding = self.id_adapter(id_uncond, id_vit_hidden_uncond)

        if num_zero !=0:
            zero_tensor = torch.zeros(
                (id_embedding.size(0), num_zero, id_embedding.size(-1)),
                dtype=id_embedding.dtype,
                device=id_embedding.device,
            )
            id_embedding = torch.cat([id_embedding, zero_tensor], dim=1)
            uncond_id_embedding = torch.cat([uncond_id_embedding, zero_tensor], dim=1)
        
        return id_embedding, uncond_id_embedding

    def get_id_embedding_project(self, id_embedding, uncond_id_embedding):
        id_embedding = torch.cat([uncond_id_embedding, id_embedding], dim=0).to(self.device, dtype=torch.bfloat16)
        print("id_embedding.shape", id_embedding.shape)

        all_embeddings = []
        for layer in self.id_adapter_attn_layers:
            if getattr(layer, "id_to_k", None) is not None:
                combined_tensor = torch.cat([
                    layer.id_to_k(id_embedding),
                    layer.id_to_v(id_embedding)
                ], dim=0)
                # print("combined_tensor.shape", combined_tensor.shape)
                # if layer.hidden_size == 640:
                combined_tensor = combined_tensor.reshape(-1, id_embedding.size(1), 1280)
                # print("combined_tensor.shape", combined_tensor.shape)
                all_embeddings.append(combined_tensor)
        all_embeddings = torch.cat(all_embeddings, dim=0).to(self.device, dtype=torch.bfloat16)
        # print("all_embeddings.shape", all_embeddings.shape)
        return all_embeddings


    def inference(
        self,
        prompt,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt='',
        id_images=[],
        id_scale=1.0,
        guidance_scale=1.2,
        num_inference_steps=4,
        seed=-1,
        num_zero=8,
        ortho='v2',
    ):
        # 预处理开始时间
        t_preprocess_start = time.time()
        
        attention.NUM_ZERO = num_zero
    
        if ortho == 'v2':
            attention.ORTHO = False
            attention.ORTHO_v2 = True
        elif ortho == 'v1':
            attention.ORTHO = True
            attention.ORTHO_v2 = False
        else:
            attention.ORTHO = False
            attention.ORTHO_v2 = False
        print("ORTHO: ", attention.ORTHO, "ORTHO_v2: ", attention.ORTHO_v2)
        id_embedding, uncond_id_embedding = self.get_id_embedding(id_images,num_zero)
        print("id_embedding.shape", id_embedding.shape)
                # 预处理结束时间
        t_preprocess_end = time.time()
        print(f"预处理时间: {t_preprocess_end - t_preprocess_start:.2f}秒")

        latents, t_generate_end = self.new_method(prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, seed, id_embedding, uncond_id_embedding, id_scale)
        
        # 解码latent到图像
        images = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = self.pipe.image_processor.postprocess(images, output_type="pil")
        
        # 解码结束时间
        t_decode_end = time.time()
        print(f"解码时间: {t_decode_end - t_generate_end:.2f}秒")
        print(f"总耗时: {t_decode_end - t_preprocess_start:.2f}秒")

        return images

    def generate_latent(self, prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, seed, id_embedding, uncond_id_embedding, id_scale,id_mask=None,**kwargs):
        simple_start = time.time()
        id_embedding = self.get_id_embedding_project(id_embedding, uncond_id_embedding)
        cross_attention_kwargs = {'id_embedding': id_embedding, 'id_scale': id_scale, 'id_mask': id_mask}       
       
        # 生成latent
        latents = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            cross_attention_kwargs=cross_attention_kwargs,
            output_type="latent",
            **kwargs,
        ).images

        # 生成latent结束时间
        t_generate_end = time.time()
        print(f"采样latent时间: {t_generate_end - simple_start:.2f}秒")
        return latents,t_generate_end
    

if __name__ == "__main__":
    pipeline = PuLIDPipeline(sdxl_repo='RunDiffusion/Juggernaut-XL-v9')
    # 读取图片并转换为RGB格式
    image = cv2.imread('pulid/test.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pipeline.inference(prompt='a beautiful girl', id_images=[image], id_scale=1.0, num_inference_steps=2, guidance_scale=7, ortho='')
    result[0].save('result3.png')

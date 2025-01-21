import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
# from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers import EulerDiscreteScheduler

from animation.modules.vae import AutoencoderKLTemporalDecoder
from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet import UNetSpatioTemporalConditionModel
from animation.pipelines.inference_pipeline_animation import InferenceAnimationPipeline
import random
import pdb
import todos

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def load_images_from_folder(folder, width, height):
    images = []
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        img = img.resize((width, height))
        images.append(img)

    return images

def save_frames_as_png(frames, output_path):
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    num_frames = len(pil_frames)
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        save_path = os.path.join(output_path, f'frame_{i}.png')
        pil_frame.save(save_path)

def save_frames_as_mp4(frames, output_mp4_path, fps):
    print("Starting saving the frames as mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'H264' for better quality
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    for frame in frames:
        frame_bgr = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=125,
                       loop=0)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--reference_image",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--reference_image`s, or a single"
            " `--reference_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--pose_control_folder",
        type=str,
        default=None,
        help=(
            "the validation control image"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--height",
        type=int,
        default=768,
        required=False
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        required=False
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        required=False
    )

    parser.add_argument(
        "--posenet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained posenet model",
    )
    parser.add_argument(
        "--face_encoder_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained face encoder model",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model",
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=16,
        required=False
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        required=False
    )

    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,  # or set to 0.02
        required=False
    )
    parser.add_argument(
        "--frames_overlap",
        type=int,
        default=4,
        required=False
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=None,
        required=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # torch.set_default_dtype(torch.float16)
    seed = 23123134
    # seed = 42
    # seed = 123
    seed_everything(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    # do_resize = True
    # size = {'shortest_edge': 224}
    # resample = 3
    # do_center_crop = True
    # crop_size = {'height': 224, 'width': 224}
    # do_rescale = True
    # rescale_factor = 0.00392156862745098
    # do_normalize = True
    # image_mean = [0.48145466, 0.4578275, 0.40821073]
    # image_std = [0.26862954, 0.26130258, 0.27577711]
    # do_convert_rgb = True
    # kwargs = {'feature_extractor_type': 'CLIPFeatureExtractor', 'image_processor_type': 'CLIPImageProcessor'}

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    # image_encoder.forward.__code__ ---
    # <file "miniconda3/envs/python39/lib/python3.9/site-packages/transformers/models/clip/modeling_clip.py", line 1500>
    # model name: CLIPVisionModelWithProjection
    # model file: checkpoints/SVD/stable-video-diffusion-img2vid-xt/image_encoder/model.safetensors

    # args.pretrained_model_name_or_path === 'checkpoints/SVD/stable-video-diffusion-img2vid-xt'
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision) # args.revision -- None
    # vae.forward.__code__ --
    #     <file "miniconda3/envs/python39/lib/python3.9/site-packages/diffusers/models/
    #     autoencoders/autoencoder_kl_temporal_decoder.py", line 373>
    # model name:  AutoencoderKLTemporalDecoder
    # model file:  checkpoints/SVD/stable-video-diffusion-img2vid-xt/vae/diffusion_pytorch_model.safetensors

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    # model name:  UNetSpatioTemporalConditionModel
    # model file:  checkpoints/SVD/stable-video-diffusion-img2vid-xt/unet/diffusion_pytorch_model.safetensors


    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0]) # 320 -- unet.config.block_out_channels[0]

    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        # clip_embeddings_dim=image_encoder.config.hidden_size,
        clip_embeddings_dim=1024,
        num_tokens=4, )
    face_model = FaceModel()
    # face_model.face_helper


    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()
    # unet.attn_processors.keys()
    # dict_keys(['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn2.processor', 
    #     'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'mid_block.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'mid_block.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'mid_block.attentions.0.temporal_transformer_blocks.0.attn2.processor'])

    for name in unet.attn_processors.keys():
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
            # name -- 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                # name -- 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'
                # hidden_size -- 320, cross_attention_dim === None
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                # name -- 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor'
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_svd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            #  name -- 'down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn1.processor'
            # cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            # if cross_attention_dim is None:
            #     attn_procs[name] = XFormersAttnProcessor()
            # else:
            #     attn_procs[name] = XFormersAttnProcessor()
            attn_procs[name] = XFormersAttnProcessor()

    # (Pdb) attn_procs.keys()
    # dict_keys(['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor', 
    #     'up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn1.processor', 
    #     'up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn2.processor', 
    #     'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 
    #     'mid_block.attentions.0.transformer_blocks.0.attn2.processor', 
    #     'mid_block.attentions.0.temporal_transformer_blocks.0.attn1.processor', 
    #     'mid_block.attentions.0.temporal_transformer_blocks.0.attn2.processor'])


    unet.set_attn_processor(attn_procs)
    # unet -- UNetSpatioTemporalConditionModel(...)

    # resume the previous checkpoint
    if args.posenet_model_name_or_path is not None and args.face_encoder_model_name_or_path is not None and args.unet_model_name_or_path is not None:
        print("Loading existing posenet weights, face_encoder weights and unet weights.")
        if args.posenet_model_name_or_path.endswith(".pth"):
            # args.posenet_model_name_or_path -- checkpoints/Animation/pose_net.pth'
            pose_net_state_dict = torch.load(args.posenet_model_name_or_path, map_location="cpu")
            pose_net.load_state_dict(pose_net_state_dict, strict=True)
        else:
            print("posenet weights loading fail")
            print(1/0)
        if args.face_encoder_model_name_or_path.endswith(".pth"):
            # args.face_encoder_model_name_or_path -- 'checkpoints/Animation/face_encoder.pth'
            face_encoder_state_dict = torch.load(args.face_encoder_model_name_or_path, map_location="cpu")
            face_encoder.load_state_dict(face_encoder_state_dict, strict=True)
        else:
            print("face_encoder weights loading fail")
            print(1/0)
        if args.unet_model_name_or_path.endswith(".pth"):
            #  args.unet_model_name_or_path -- 'checkpoints/Animation/unet.pth'
            unet_state_dict = torch.load(args.unet_model_name_or_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=True)
        else:
            print("unet weights loading fail")
            print(1/0)

    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.float16
    # weight_dtype = torch.float32
    # weight_dtype = torch.bfloat16

    # vae -- AutoencoderKLTemporalDecoder()
    # vae.encoder -- Encoder
    # vae.decoder -- Decoder
    # ver.quant_conv -- Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))

    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device='cuda', dtype=weight_dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    # args.reference_image -- 'inference/case-1/reference.png'
    reference_image_path = args.reference_image
    reference_image = Image.open(args.reference_image).convert('RGB')
    # pp args.pose_control_folder -- 'inference/case-1/poses'
    pose_control_images = load_images_from_folder(args.pose_control_folder, width=args.width, height=args.height)

    num_frames = len(pose_control_images) # 16
    face_model.face_helper.clean_all()

    reference_face = cv2.imread(reference_image_path)
    validation_image_bgr = cv2.cvtColor(reference_face, cv2.COLOR_RGB2BGR)
    reference_image_face_info = face_model.app.get(validation_image_bgr)
    # reference_image_face_info is dict:
    #     array [bbox] shape: (4,), min: 10.485265731811523, max: 309.0953674316406, mean: 164.8896484375
    #     array [kps] shape: (5, 2), min: 37.669029235839844, max: 289.92230224609375, mean: 164.64889526367188
    #     [det_score] type: <class 'numpy.float32'>
    #     array [landmark_3d_68] shape: (68, 3), min: -2.8244519233703613, max: 312.5506896972656, mean: 114.32061004638672
    #     array [pose] shape: (3,), min: -5.829032897949219, max: 8.235852241516113, mean: -0.47325700521469116
    #     array [landmark_2d_106] shape: (106, 2), min: 27.393348693847656, max: 307.9230651855469, mean: 164.82591247558594
    #     [gender] type: <class 'numpy.int64'>
    #     [age] value: '36'
    #     array [embedding] shape: (512,), min: -3.173130989074707, max: 2.986737012863159, mean: 0.009902000427246094

    if len(reference_image_face_info) > 0:
        reference_image_face_info = sorted(reference_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        reference_image_id_ante_embedding = reference_image_face_info['embedding']
    else:
        reference_image_id_ante_embedding = None
        # reference_image_id_ante_embedding = np.zeros((512,))

    # generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    decode_chunk_size = args.decode_chunk_size # 4
    video_frames = pipeline(
        image=reference_image,
        image_pose=pose_control_images,
        height=args.height,
        width=args.width,
        num_frames=num_frames, # 16
        tile_size=args.tile_size,
        tile_overlap=args.frames_overlap,
        decode_chunk_size=decode_chunk_size,
        motion_bucket_id=127.,
        fps=7,
        min_guidance_scale=args.guidance_scale,
        max_guidance_scale=args.guidance_scale,
        noise_aug_strength=args.noise_aug_strength,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output_type="pil",
        reference_image_id_ante_embedding=reference_image_id_ante_embedding, # shape -- (512,)
    )[0]
    out_file = os.path.join(
        args.output_dir,
        f"animation_video.mp4",
    )
    for i in range(num_frames):
        img = video_frames[i]
        video_frames[i] = np.array(img)

    png_out_file = os.path.join(args.output_dir, "animated_images")
    os.makedirs(png_out_file, exist_ok=True)
    export_to_gif(video_frames, out_file, 8)
    save_frames_as_png(video_frames, png_out_file)


# bash command_basic_infer.sh

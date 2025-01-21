import inspect
# from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import PIL.Image
# import einops
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor


from einops import rearrange
import pdb
import todos

from diffusers.utils import logging
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: List[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    # tensor [tensor2vid video] size: [1, 3, 16, 512, 512], min: -1.197266, max: 1.200195, mean: -0.009327
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)

    # tensor2vid outputs is list: len = 1
    #     [item] type: <class 'list'>
    # tensor2vid outputs[0] is list: len = 16
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>
    #     [item] type: <class 'PIL.Image.Image'>

    return outputs

class InferenceAnimationPipeline(DiffusionPipeline):
    def __init__(self,
            vae,
            image_encoder,
            unet,
            scheduler,
            feature_extractor,
            pose_net,
            face_encoder,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net,
            face_encoder=face_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) # 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.num_tokens = 4

    def _encode_image(self,
            image,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool):
        # num_videos_per_prompt = 1
        # do_classifier_free_guidance = True
        dtype = next(self.image_encoder.parameters()).dtype # torch.float16
        # [_encode_image image] type: <class 'PIL.Image.Image'>

        if not isinstance(image, torch.Tensor): # True ?
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values
            # tensor [image] size: [1, 3, 224, 224], min: -1.799994, max: 2.17203, mean: 0.099575
        else:
            pdb.set_trace()

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # tensor [_encode_image image_embeddings] size: [1, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285
        if do_classifier_free_guidance: # True
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        # tensor [image_embeddings] size: [2, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.002143
        return image_embeddings

    def _encode_vae_image(self,
            image: torch.Tensor,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        # num_videos_per_prompt = 1
        # do_classifier_free_guidance = True
        # tensor [image] size: [1, 3, 512, 512], min: -1.085592, max: 1.066735, mean: -0.045391

        image = image.to(device=device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(image).latent_dist.mode()
        # tensor [image_latents] size: [1, 4, 64, 64], min: -33.178013, max: 35.71368, mean: -1.375864

        if do_classifier_free_guidance: # True
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
        # tensor [image_latents] size: [2, 4, 64, 64], min: -33.178013, max: 35.71368, mean: -0.687932

        return image_latents

    def _get_add_time_ids(self,
            fps: int,
            motion_bucket_id: int,
            noise_aug_strength: float,
            dtype: torch.dtype,
            batch_size: int,
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        # fps = 6
        # motion_bucket_id = 127.0
        # noise_aug_strength = 0.02
        # dtype = torch.float16
        # batch_size = 1
        # num_videos_per_prompt = 1
        # do_classifier_free_guidance = True

        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        # todos.debug.output_var("_get_add_time_ids add_time_ids", add_time_ids)
        # _get_add_time_ids add_time_ids is list: len = 3
        #     [item] value: '6'
        #     [item] value: '127.0'
        #     [item] value: '0.02'

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features # 768

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, " \
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. " \
                f"Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        # todos.debug.output_var("_get_add_time_ids add_time_ids", add_time_ids)
        # tensor [_get_add_time_ids add_time_ids] size: [1, 3], min: 0.020004, max: 127.0, mean: 44.34

        if do_classifier_free_guidance: # True
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        # tensor [add_time_ids] size: [2, 3], min: 0.020004, max: 127.0, mean: 44.340008
        return add_time_ids

    def decode_latents(self,
            latents: torch.Tensor,
            num_frames: int,
            decode_chunk_size: int = 8):
        # tensor [decode_latents latents] size: [1, 16, 4, 64, 64], min: -6.613281, max: 7.503906, mean: -0.16142
        # num_frames = 16
        # decode_chunk_size = 4

        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents # self.vae.config.scaling_factor -- 0.18215

        # forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        # accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
        # assert accepts_num_frames == True

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        # latents.shape[0] === 16
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            # if accepts_num_frames: # True
            #     # we only pass num_frames_in if it's expected
            #     decode_kwargs["num_frames"] = num_frames_in
            decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame.cpu())
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        # tensor [decode_latents frames] size: [1, 3, 16, 512, 512], min: -1.197266, max: 1.200195, mean: -0.009327

        return frames

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(self,
            batch_size: int,
            num_frames: int,
            num_channels_latents: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            generator: torch.Generator,
            latents: Optional[torch.Tensor] = None,
    ):

        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor, # 8
            width // self.vae_scale_factor, # 8
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma 
        # self.scheduler.init_noise_sigma === tensor(700.000732)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
            self,
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
            image_pose: Union[torch.FloatTensor],
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            tile_size: Optional[int] = 16,
            tile_overlap: Optional[int] = 4,
            num_inference_steps: int = 25,
            min_guidance_scale: float = 1.0,
            max_guidance_scale: float = 3.0,
            fps: int = 7,
            motion_bucket_id: int = 127,
            noise_aug_strength: float = 0.02,
            image_only_indicator: bool = False,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            reference_image_id_ante_embedding=None,
            output_type: Optional[str] = "pil",
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            return_dict: bool = True,
    ):
        # height = 512
        # width = 512
        # num_frames = 16
        # tile_size = 16
        # tile_overlap = 4
        # num_inference_steps = 25
        # min_guidance_scale = 3.0
        # max_guidance_scale = 3.0
        # fps = 6
        # motion_bucket_id = 127.0
        # noise_aug_strength = 0.02
        # image_only_indicator = False
        # decode_chunk_size = 4
        # num_videos_per_prompt = 1
        # generator = <torch._C.Generator object at 0x7f09d0a457d0>
        # latents = None
        # reference_image_id_ante_embedding = tensor([[ 0.770020, -0.522461,  0.873535,  ..., -0.445557,  0.213135,
        #          -1.483398]], device='cuda:0', dtype=torch.float16)
        # output_type = 'pil'
        # callback_on_step_end = None
        # callback_on_step_end_tensor_inputs = ['latents']
        # return_dict = True


        # 0. Default height and width to unet
        # self.unet.config.sample_size === 96, self.vae_scale_factor === 8
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale >= 1.0
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        fps = fps - 1

        # 4. Encode input image using VAE

        # print(image_embeddings.size()) # [2, 1, 1024]
        reference_image_id_ante_embedding = torch.from_numpy(reference_image_id_ante_embedding).unsqueeze(0)
        reference_image_id_ante_embedding = reference_image_id_ante_embedding.to(device=device, dtype=image_embeddings.dtype)

        faceid_latents = self.face_encoder(reference_image_id_ante_embedding, image_embeddings[1:])
        # tensor [faceid_latents] size: [1, 4, 1024], min: -14.492188, max: 14.453125, mean: 3.8e-05

        uncond_image_embeddings = image_embeddings[:1]
        uncond_faceid_latents = torch.zeros_like(faceid_latents)
        uncond_image_embeddings = torch.cat([uncond_image_embeddings, uncond_faceid_latents], dim=1)
        cond_image_embeddings = image_embeddings[1:]
        cond_image_embeddings = torch.cat([cond_image_embeddings, faceid_latents], dim=1)
        image_embeddings = torch.cat([uncond_image_embeddings, cond_image_embeddings])

        image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise # noise_aug_strength === 0.02
        # tensor [image] size: [1, 3, 512, 512], min: -1.085592, max: 1.066735, mean: -0.045391

        needs_upcasting = (self.vae.dtype == torch.float16 or self.vae.dtype == torch.bfloat16) and self.vae.config.force_upcast
        if needs_upcasting: # True
            self_vae_dtype = self.vae.dtype
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance, # True
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        if needs_upcasting: # True
            self.vae.to(dtype=self_vae_dtype)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        # tensor [image_latents] size: [2, 16, 4, 64, 64], min: -33.1875, max: 35.71875, mean: -0.687928

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels # 8

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            tile_size,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents = latents.repeat(1, num_frames // tile_size + 1, 1, 1, 1)[:, :num_frames]
        # tensor [latents] size: [1, 16, 4, 64, 64], min: -6.613281, max: 7.503906, mean: -0.16142

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale # === 3.0

        # 8. Denoising loop
        self._num_timesteps = len(timesteps)
        indices = [[0, *range(i + 1, min(i + tile_size, num_frames))] for i in
                   range(0, num_frames - tile_size + 1, tile_size - tile_overlap)]
        if indices[-1][-1] < num_frames - 1:
            indices.append([0, *range(num_frames - tile_size + 1, num_frames)])

        pose_pil_image_list = []
        for pose in image_pose:
            pose = torch.from_numpy(np.array(pose)).float()
            pose = pose / 127.5 - 1
            pose_pil_image_list.append(pose)
        pose_pil_image_list = torch.stack(pose_pil_image_list, dim=0)
        pose_pil_image_list = rearrange(pose_pil_image_list, "f h w c -> f c h w")
        # tensor [pose_pil_image_list] size: [16, 3, 512, 512], min: -1.0, max: 1.0, mean: -0.994078

        # print(indices)  # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
        self.pose_net.to(device)
        self.unet.to(device)

        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        # len(timesteps) -- 25
        # timesteps
        # tensor([ 1.637770,  1.575531,  1.510996,  1.443990,  1.374316,  1.301752,
        #          1.226049,  1.146922,  1.064048,  0.977053,  0.885506,  0.788904,
        #          0.686657,  0.578063,  0.462282,  0.338294,  0.204848,  0.060379,
        #         -0.097098, -0.270160, -0.462234, -0.678018, -0.924202, -1.210778,
        #         -1.553652], device='cuda:0')

        with self.progress_bar(total=len(timesteps) * len(indices)) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # tensor [latent_model_input] size: [2, 16, 4, 64, 64], min: -6.613281, max: 7.503906, mean: -0.161417

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                # tensor [latent_model_input] size: [2, 16, 8, 64, 64], min: -33.1875, max: 35.71875, mean: -0.424673

                # predict the noise residual
                noise_pred = torch.zeros_like(image_latents)
                noise_pred_cnt = image_latents.new_zeros((num_frames,))
                weight = (torch.arange(tile_size, device=device) + 0.5) * 2. / tile_size
                weight = torch.minimum(weight, 2 - weight)
                # indices -- [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
                for idx in indices:
                    # classification-free inference
                    pose_latents = self.pose_net(pose_pil_image_list[idx].to(device=device, dtype=latent_model_input.dtype))
                    # tensor [pose_latents] size: [16, 320, 64, 64], min: -1.385742, max: 1.225586, mean: 0.00031
                    _noise_pred = self.unet(
                        latent_model_input[:1, idx],
                        t,
                        encoder_hidden_states=image_embeddings[:1],
                        added_time_ids=added_time_ids[:1],
                        pose_latents=None,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]
                    # tensor [_noise_pred] size: [1, 16, 4, 64, 64], min: -1.219727, max: 1.305664, mean: 0.001827
                    noise_pred[:1, idx] += _noise_pred * weight[:, None, None, None]

                    # normal inference
                    _noise_pred = self.unet(
                        latent_model_input[1:, idx],
                        t,
                        encoder_hidden_states=image_embeddings[1:],
                        added_time_ids=added_time_ids[1:],
                        pose_latents=pose_latents,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]
                    noise_pred[1:, idx] += _noise_pred * weight[:, None, None, None]

                    noise_pred_cnt[idx] += weight
                    progress_bar.update()
                noise_pred.div_(noise_pred_cnt[:, None, None, None])
                # tensor [noise_pred] size: [2, 16, 4, 64, 64], min: -2.486328, max: 2.761719, mean: -0.001121

                # perform guidance
                # self.do_classifier_free_guidance -- tensor(True, device='cuda:0')
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        self.vae.decoder.to(device)
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        # print(frames.size()) # [1, 3, 16, 512, 512]
        # print(latents.size()) # [1, 16, 4, 64, 64]
        frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        # print(frames[0].size()) # [16, 3, 512, 512]

        self.maybe_free_model_hooks()

        return frames

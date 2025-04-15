import warnings
import inspect
import torch

from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
from functools import partial


##############################################
# Adapted SANA inference code from Diffusers #
##############################################

# https://github.com/huggingface/diffusers/blob/4b868f14c15cddee610d130be3c65a37b6793285/src/diffusers/pipelines/sana/pipeline_sana.py#L704



def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
def _predict_fn(latents, timestep,
                transformer,
                prompt_embeds, prompt_attention_mask,
                attention_kwargs,
                do_classifier_free_guidance, guidance_scale,
                self):
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = latent_model_input.to(prompt_embeds.dtype)

    if len(timestep) != len(latent_model_input):
        timestep = timestep.expand(latent_model_input.shape[0]).to(latents.dtype)

    noise_pred = transformer(
        latent_model_input,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=timestep,
        return_dict=False,
        attention_kwargs=attention_kwargs,
    )[0]
    noise_pred = noise_pred.float()
    self.n_steps += 1

    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred


@torch.no_grad()
def run(
    self,
    step_fn,
    prompt: Union[str, List[str]] = None,
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 4.5,
    num_images_per_prompt: Optional[int] = 1,
    height: int = 1024,
    width: int = 1024,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    clean_caption: bool = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 300,
    complex_human_instruction: List[str] = [
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ],
) -> Union[SanaPipelineOutput, Tuple]:
    
    self.check_inputs(
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._interrupt = False

    # 2. Default height and width to transformer
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    lora_scale = self.attention_kwargs.get("scale", None) if self.attention_kwargs is not None else None

    # 3. Encode input prompt
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = self.encode_prompt(
        prompt,
        self.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        clean_caption=clean_caption,
        max_sequence_length=max_sequence_length,
        complex_human_instruction=complex_human_instruction,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    # 4. Prepare timesteps
    if sigmas is None:
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
    else:
        sigmas = torch.tensor(sigmas).to('cuda')
        timesteps = (sigmas[:-1] * 1000).to(int)
        num_inference_steps = len(timesteps)
        self.scheduler.sigmas = sigmas

    # 5. Prepare latents.
    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        latent_channels,
        height,
        width,
        torch.float32,
        device,
        generator,
        latents,
    )

    # 6. Denoising loop
    self._num_timesteps = len(timesteps)
    cache = []
    self.n_steps = 0

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = latent_model_input.to(prompt_embeds.dtype)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=timestep,
                return_dict=False,
                attention_kwargs=self.attention_kwargs,
            )[0]
            noise_pred = noise_pred.float()
            self.n_steps += 1

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # compute previous image: x_t -> x_t-1
            v_pred_fn = partial(_predict_fn,
                                timestep=timestep,
                                transformer=self.transformer,
                                prompt_embeds=prompt_embeds,
                                prompt_attention_mask=prompt_attention_mask,
                                attention_kwargs=self.attention_kwargs,
                                do_classifier_free_guidance=self.do_classifier_free_guidance,
                                guidance_scale=guidance_scale,
                                self=self
                                )
            eps = latents + (1 - self.scheduler.sigmas[i]) * noise_pred
            latents = step_fn(latents, self.scheduler.sigmas, noise_pred, i,
                              v_pred_fn=v_pred_fn, cache=cache)
            if len(cache) > 0:
                cache.pop()
            cache.append(eps)

            # call the callback, if provided
            progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        latents = latents.to(self.vae.dtype)
        try:
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        except torch.cuda.OutOfMemoryError as e:
            warnings.warn(
                f"{e}. \n"
                f"Try to use VAE tiling for large images. For example: \n"
                f"pipe.vae.enable_tiling(tile_sample_min_width=512, tile_sample_min_height=512)"
            )
            
    if not output_type == "latent":
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    print(f'The number of neural network evaluations (NFE) is equal to {self.n_steps}.')

    return SanaPipelineOutput(images=image)

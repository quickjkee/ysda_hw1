
import torch

from typing import Optional
from functools import partial


###############################################
# Adapted SD1.5 inference code from Diffusers #
###############################################

# https://github.com/huggingface/diffusers/blob/4b868f14c15cddee610d130be3c65a37b6793285/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L780 


@torch.no_grad()
def _predict_fn(
    unet,
    latents, 
    timestep,
    prompt_embeds,
    do_classifier_free_guidance, 
    guidance_scale,
):
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = latent_model_input.to(prompt_embeds.dtype)

    if len(timestep) != len(latent_model_input):
        timestep = timestep.expand(latent_model_input.shape[0]).to(latents.dtype)

    u_pred = unet(
        latent_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=timestep,
        return_dict=False,
    )[0]
    u_pred = u_pred.float()

    # perform guidance
    if do_classifier_free_guidance:
        u_pred_uncond, u_pred_text = u_pred.chunk(2)
        u_pred = u_pred_uncond + guidance_scale * (u_pred_text - u_pred_uncond)

    return u_pred



@torch.no_grad()
def run(
    pipe,
    prompt,
    step_fn,
    sigmas,
    height = 512,
    width = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt = 1,
    generator = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds = None,
    device = 'cuda'
):  
    # 2. Define call parameters
    if isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
     
    sigmas = torch.tensor(sigmas).to('cuda')
    timesteps = (sigmas[:-1] * 1000).to(int)
    num_inference_steps = len(timesteps)
    pipe.scheduler.sigmas = sigmas
   
    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latent_shape = (
        batch_size * num_images_per_prompt,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    latents = torch.randn(
        latent_shape,
        generator=generator, 
        device=device, 
        dtype=prompt_embeds.dtype
    )
    cache = []
    
    # 6. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            u_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                u_pred_uncond, u_pred_text = u_pred.chunk(2)
                u_pred = u_pred_uncond + guidance_scale * (u_pred_text - u_pred_uncond)


            v_pred_fn = partial(
                _predict_fn,
                unet=pipe.unet,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale
            )
            
            # compute the previous noisy sample x_t -> x_t-1
            eps = latents + (1 - pipe.scheduler.sigmas[i]) * u_pred
            latents = step_fn(
                latents, pipe.scheduler.sigmas, u_pred, i,
                v_pred_fn=v_pred_fn, cache=cache
            )
            
            if len(cache) > 0:
                cache.pop()
            cache.append(eps)
            
            # call the callback, if provided
            progress_bar.update()
                
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[0]    
    do_denormalize = [True] * image.shape[0]
    image = pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)
    return image

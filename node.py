import gc
import html
import os
import re
import torch
import numpy as np
from pathlib import Path

import folder_paths

NODE_DIR    = Path(__file__).parent
CONFIGS_DIR = NODE_DIR / "walkyrie_configs"

_loaded_cache = {}


def _split_state_dict(merged_sd):
    components = {"transformer": {}, "text_encoder": {}, "vae": {}}
    for k, v in merged_sd.items():
        for comp in components:
            if k.startswith(comp + "."):
                components[comp][k[len(comp) + 1:]] = v
                break
    return components


def _prompt_clean(text):
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _encode_prompt(tokenizer, text_encoder, prompts, max_length, device, dtype):
    prompts = [_prompt_clean(p) for p in prompts]
    tokens  = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids      = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    seq_lens       = attention_mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        embeds = text_encoder(input_ids, attention_mask).last_hidden_state.to(dtype)

    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_length - u.size(0), u.size(1))]) for u in embeds], dim=0
    )
    return embeds


def _unload_cached_models():
    global _loaded_cache
    if not _loaded_cache:
        return
    print("[Walkyrie] Clearing previous model from memory...")
    for key, models in _loaded_cache.items():
        for name in ("text_encoder", "vae", "transformer"):
            m = models.get(name)
            if m is not None:
                m.to("cpu")
                del m
    _loaded_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Walkyrie] Memory cleared (RAM + VRAM).")


def _load_models(configs_dir, merged_file, torch_dtype):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, UMT5EncoderModel, AutoConfig
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, FlowMatchEulerDiscreteScheduler

    cache_key = (merged_file, torch_dtype)

    if cache_key in _loaded_cache:
        print("[Walkyrie] Using cached models.")
        return _loaded_cache[cache_key]

    if _loaded_cache:
        _unload_cached_models()

    print(f"[Walkyrie] Loading: {merged_file}")
    merged_sd = load_file(merged_file, device="cpu")
    sd        = _split_state_dict(merged_sd)

    tokenizer    = AutoTokenizer.from_pretrained(configs_dir, subfolder="tokenizer")

    te_config    = AutoConfig.from_pretrained(os.path.join(configs_dir, "text_encoder"))
    text_encoder = UMT5EncoderModel(te_config).to(torch_dtype)
    text_encoder.load_state_dict(sd["text_encoder"], strict=False)
    text_encoder.eval()

    vae_config = AutoencoderKLWan.load_config(configs_dir, subfolder="vae")
    vae        = AutoencoderKLWan.from_config(vae_config).to(torch_dtype)
    vae.load_state_dict(sd["vae"], strict=False)
    vae.eval()

    tr_config   = WanTransformer3DModel.load_config(configs_dir, subfolder="transformer")
    transformer = WanTransformer3DModel.from_config(tr_config).to(torch_dtype)
    transformer.load_state_dict(sd["transformer"], strict=False)
    transformer.eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(configs_dir, subfolder="scheduler")

    models = dict(
        tokenizer    = tokenizer,
        text_encoder = text_encoder,
        vae          = vae,
        transformer  = transformer,
        scheduler    = scheduler,
    )
    _loaded_cache[cache_key] = models
    print("[Walkyrie] Models ready.")
    return models


@torch.no_grad()
def _run_inference(models, prompt, height, width, steps, guidance_scale, generator, device, dtype):
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    from comfy.utils import ProgressBar

    tokenizer    = models["tokenizer"]
    text_encoder = models["text_encoder"].to(device)
    vae          = models["vae"].to(device)
    transformer  = models["transformer"].to(device)
    scheduler    = models["scheduler"]

    prompts     = [prompt] if isinstance(prompt, str) else prompt
    batch_size  = len(prompts)

    prompt_embeds   = _encode_prompt(tokenizer, text_encoder, prompts,            226, device, dtype)
    negative_embeds = _encode_prompt(tokenizer, text_encoder, [""] * batch_size, 226, device, dtype)

    vae_scale   = getattr(vae.config, "scale_factor_spatial", 8)
    in_channels = transformer.config.in_channels
    shape       = (batch_size, in_channels, 1, height // vae_scale, width // vae_scale)
    latents     = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    comfy_pbar = ProgressBar(steps)
    cli_iter   = (
        tqdm(timesteps, desc="[Walkyrie] Sampling", unit="step", dynamic_ncols=True)
        if _has_tqdm
        else timesteps
    )

    for i, t in enumerate(cli_iter):
        latents_input   = torch.cat([latents, latents])
        combined_embeds = torch.cat([negative_embeds, prompt_embeds])

        noise_pred = transformer(
            hidden_states=latents_input.to(dtype),
            timestep=t.expand(batch_size * 2),
            encoder_hidden_states=combined_embeds,
            return_dict=False,
        )[0]

        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        comfy_pbar.update(1)
        if not _has_tqdm:
            print(f"[Walkyrie] Step {i + 1}/{steps}")

    latents      = latents.to(vae.dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents)
    latents_std  = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents)
    latents      = latents / latents_std + latents_mean

    image = vae.decode(latents, return_dict=False)[0]
    image = image.squeeze(2)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float()
    return image


class WalkyrieNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name"     : (folder_paths.get_filename_list("checkpoints"),),
                "prompt"        : ("STRING", {"multiline": True, "default": ""}),
                "height"        : ("INT",   {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "width"         : ("INT",   {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "steps"         : ("INT",   {"default": 20,  "min": 1,  "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed"          : ("INT",   {"default": 42,  "min": 0,  "max": 0xffffffffffffffff}),
                "dtype"         : (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "generate"
    CATEGORY     = "Walkyrie"

    def generate(self, ckpt_name, prompt, height, width, steps, guidance_scale, seed, dtype):
        dtype_map   = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map[dtype]
        device      = "cuda" if torch.cuda.is_available() else "cpu"

        merged_file = folder_paths.get_full_path("checkpoints", ckpt_name)
        models      = _load_models(str(CONFIGS_DIR), merged_file, torch_dtype)
        generator   = torch.Generator(device=device).manual_seed(seed)

        image = _run_inference(models, prompt, height, width, steps, guidance_scale, generator, device, torch_dtype)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "WalkyrieNode": WalkyrieNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WalkyrieNode": "Walkyrie",
}
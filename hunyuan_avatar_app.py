import os
import time
import argparse
import gradio as gr
import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import tempfile
import shutil
import gc
import traceback
from moviepy.editor import ImageSequenceClip
from hyvideo.hunyuan import HunyuanVideoSampler
from hyvideo.modules.models import get_linear_split_map
from importlib.metadata import version
#import offload
from wan.utils.utils import cache_video

from mmgp import offload

from wan.modules.attention import get_attention_modes, get_supported_attention_modes

attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()

def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

offload.shared_state["_attention"] = get_auto_attention()

# Constants
WanGP_version = "6.2"
settings_version = 2
AUTOSAVE_FILENAME = "queue.zip"
PROMPT_VARS_MAX = 10

# Model configurations
model_types = ["hunyuan_avatar"]
model_signatures = {"hunyuan_avatar": "hunyuan_video_avatar"}
transformer_choices = ["ckpts/hunyuan_video_avatar_720_bf16.safetensors", "ckpts/hunyuan_video_avatar_720_quanto_bf16_int8.safetensors"]

# Global variables
wan_model = None
offloadobj = None
transformer_type = "hunyuan_avatar"
transformer_quantization = "int8"
transformer_dtype_policy = ""
text_encoder_quantization = "int8"
attention_mode = "auto"
compile = ""
profile = 2
vae_config = 0
boost = 1
save_path = "outputs"
preload_model_policy = []
server_config = {
    "attention_mode": "auto",
    "transformer_types": [],
    "transformer_quantization": "int8",
    "text_encoder_quantization": "int8",
    "save_path": "outputs",
    "compile": "",
    "profile": 2,
    "vae_config": 0,
    "boost": 1,
    "preload_model_policy": [],
}

# Functions
def get_model_name(model_type):
    return "Hunyuan Video Avatar 720p 13B"

def get_model_filename(model_type, quantization="int8", dtype_policy=""):
    if quantization == "int8":
        return "ckpts/hunyuan_video_avatar_720_quanto_bf16_int8.safetensors"
    else:
        return "ckpts/hunyuan_video_avatar_720_bf16.safetensors"

def get_model_type(model_filename):
    for model_type, signature in model_signatures.items():
        if signature in model_filename:
            return model_type
    return None

def get_base_model_type(model_type):
    return model_type

def get_model_family(model_type):
    return "hunyuan"

def load_hunyuan_model(model_filename, model_type=None, base_model_type=None, quantizeTransformer=False, dtype=torch.bfloat16, VAE_dtype=torch.float32, mixed_precision_transformer=False, save_quantized=False):
    hunyuan_model = HunyuanVideoSampler.from_pretrained(
        model_filepath=model_filename,
        model_type=model_type,
        base_model_type=base_model_type,
        text_encoder_filepath="ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_quanto_int8.safetensors",
        dtype=dtype,
        quantizeTransformer=quantizeTransformer,
        VAE_dtype=VAE_dtype,
        mixed_precision_transformer=mixed_precision_transformer,
        save_quantized=save_quantized
    )

    pipe = {
        "transformer": hunyuan_model.model,
        "text_encoder": hunyuan_model.text_encoder,
        "text_encoder_2": hunyuan_model.text_encoder_2,
        "vae": hunyuan_model.vae
    }

    if hunyuan_model.wav2vec is not None:
        pipe["wav2vec"] = hunyuan_model.wav2vec

    split_linear_modules_map = get_linear_split_map()
    hunyuan_model.model.split_linear_modules_map = split_linear_modules_map
    offload.split_linear_modules(hunyuan_model.model, split_linear_modules_map)

    return hunyuan_model, pipe

def load_models(model_type):
    global wan_model, offloadobj
    model_filename = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)
    wan_model, pipe = load_hunyuan_model(model_filename, model_type, get_base_model_type(model_type))
    kwargs = {"extraModelsToQuantize": None}
    if profile in (2, 4, 5):
        kwargs["budgets"] = {"transformer": 100, "text_encoder": 100, "*": 3000}

    offloadobj = offload.profile(pipe, profile_no=profile, compile=compile, quantizeTransformer=False, loras="transformer", coTenantsMap={}, **kwargs)
    pipe["transformer"].enable_cache = False
    return wan_model, offloadobj, pipe["transformer"]

def generate_video(image, text, audio, progress=gr.Progress()):
    global wan_model, offloadobj

    if wan_model is None:
        wan_model, offloadobj, _ = load_models(transformer_type)

    wan_model._interrupt = False

    model_filename = get_model_filename(transformer_type, transformer_quantization, transformer_dtype_policy)
    model_type = get_model_type(model_filename)

    # Preprocess inputs
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    audio_path = audio
    prompt = text

    # Generate video
    try:
        samples = wan_model.generate(
            input_prompt=prompt,
            input_ref_images=[image],
            audio_guide=audio_path,
            frame_num=129,  # Default frame number for Hunyuan Avatar
            height=720,
            width=720,
            fit_into_canvas=True,
            sampling_steps=30,
            guide_scale=7.5,
            seed=42,
            audio_cfg_scale=7.5,
            fps=25,
        )
    except Exception as e:
        raise gr.Error(f"Error generating video: {str(e)}")

    # Save video
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
    save_prompt = prompt[:50].strip()
    file_name = f"{time_flag}_seed42_{save_prompt}.mp4"
    video_path = os.path.join(save_path, file_name)

    if audio_path is None:
        cache_video(tensor=samples[None], save_file=video_path, fps=25, nrow=1, normalize=True, value_range=(-1, 1))
    else:
        save_path_tmp = video_path[:-4] + "_tmp.mp4"
        cache_video(tensor=samples[None], save_file=save_path_tmp, fps=25, nrow=1, normalize=True, value_range=(-1, 1))
        final_command = [
            "ffmpeg", "-y", "-i", save_path_tmp, "-i", audio_path, "-c:v", "libx264", "-c:a", "aac", "-shortest",
            "-loglevel", "warning", "-nostats", video_path,
        ]
        import subprocess
        subprocess.run(final_command, check=True)
        os.remove(save_path_tmp)

    return video_path

def create_ui():
    with gr.Blocks(title="Hunyuan Avatar i2v") as demo:
        gr.Markdown("<div align=center><H1>Hunyuan Avatar i2v</H1></div>")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Input Image", type="numpy")
                text_input = gr.Textbox(label="Input Text", lines=3)
                audio_input = gr.Audio(label="Input Audio", type="filepath")
                generate_btn = gr.Button("Generate Video")
            with gr.Column():
                video_output = gr.Video(label="Generated Video")

        generate_btn.click(
            fn=generate_video,
            inputs=[image_input, text_input, audio_input],
            outputs=[video_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share = True)


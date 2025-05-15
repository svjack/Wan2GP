# Wan2.1 GP


<p align="center">
    💜 <a href=""><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.1">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper (Coming soon)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wanxai.com">Blog</a> &nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat Group</a>&nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://discord.gg/p5XbdQV7">Discord</a>&nbsp&nbsp
<br>

## 
```bash
conda activate system
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1

pip install -r requirements.txt

pip uninstall torch torchvision -y
pip install torch==2.5.0 torchvision 
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir Wan2.1-VACE-1.3B --repo-type model

python generate.py --task vace-1.3B --size "832*480" --ckpt_dir ./Wan2.1-VACE-1.3B --src_ref_images examples/girl.png,examples/snake.png --prompt "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
```

-----
<p align="center">
<b>Wan2.1 GP by DeepBeepMeep based on Wan2.1's Alibaba: Open and Advanced Large-Scale Video Generative Models for the GPU Poor</b>
</p>

In this repository, we present **Wan2.1**, a comprehensive and open suite of video foundation models that pushes the boundaries of video generation. **Wan2.1** offers these key features:
- 👍 **SOTA Performance**: **Wan2.1** consistently outperforms existing open-source models and state-of-the-art commercial solutions across multiple benchmarks.
- 👍 **Supports Consumer-grade GPUs**: The T2V-1.3B model requires only 8.19 GB VRAM, making it compatible with almost all consumer-grade GPUs. It can generate a 5-second 480P video on an RTX 4090 in about 4 minutes (without optimization techniques like quantization). Its performance is even comparable to some closed-source models.
- 👍 **Multiple Tasks**: **Wan2.1** excels in Text-to-Video, Image-to-Video, Video Editing, Text-to-Image, and Video-to-Audio, advancing the field of video generation.
- 👍 **Visual Text Generation**: **Wan2.1** is the first video model capable of generating both Chinese and English text, featuring robust text generation that enhances its practical applications.
- 👍 **Powerful Video VAE**: **Wan-VAE** delivers exceptional efficiency and performance, encoding and decoding 1080P videos of any length while preserving temporal information, making it an ideal foundation for video and image generation.


## 🔥 Latest News!!

* Mar 03, 2025: 👋 Wan2.1GP by DeepBeepMeep v1 brings: 
    - Support for all Wan including the Image to Video model
    - Reduced memory consumption by 2, with possiblity to generate more than 10s of video at 720p with a RTX 4090 and 10s of video at 480p with less than 12GB of VRAM. Many thanks to REFLEx (https://github.com/thu-ml/RIFLEx) for their algorithm that allows generating nice looking video longer than 5s.
    - The usual perks: web interface, multiple generations, loras support, sage attebtion, auto download of models, ...

* Feb 25, 2025: 👋 We've released the inference code and weights of Wan2.1.
* Feb 27, 2025: 👋 Wan2.1 has been integrated into [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/wan/). Enjoy!


## Features
*GPU Poor version by **DeepBeepMeep**. This great video generator can now run smoothly on any GPU.*

This version has the following improvements over the original Alibaba model:
- Reduce greatly the RAM requirements and VRAM requirements 
- Much faster thanks to compilation and fast loading / unloading
- Multiple profiles in order to able to run the model at a decent speed on a low end consumer config (32 GB of RAM and 12 VRAM) and to run it at a very good speed on a high end consumer config (48 GB of RAM and 24 GB of VRAM)
- Autodownloading of the needed model files
- Improved gradio interface with progression bar and more options
- Multiples prompts / multiple generations per prompt
- Support multiple pretrained Loras with 32 GB of RAM or less
- Much simpler installation


This fork by DeepBeepMeep is an integration of the mmpg module on the original model

It is an illustration on how one can set up on an existing model some fast and properly working CPU offloading with changing only a few lines of code in the core model.

For more information on how to use the mmpg module, please go to: https://github.com/deepbeepmeep/mmgp

You will find the original Wan2.1 Video repository here: https://github.com/Wan-Video/Wan2.1

 


## Installation Guide for Linux and Windows


This app has been tested on Python 3.10 / 2.6.0  / Cuda 12.4.\


```shell
sudo apt-get update && sudo apt-get install git-lfs cbm ffmpeg

# 0 Create a Python 3.10.9 environment or a venv using python
#conda create -name Wan2GP python==3.10.9  #if you have conda
conda install python==3.10.9
pip install ipykernel

#git clone https://github.com/svjack/Wan2GP && cd Wan2GP
git clone https://github.com/deepbeepmeep/Wan2GP && cd Wan2GP

# 1 Install pytorch 2.6.0
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124  

# 2. Install pip dependencies
python -m pip install -r requirements.txt
pip install "httpx[socks]"

# 3.1 optional Sage attention support (30% faster, easy to install on Linux but much harder on Windows)
python -m pip install sageattention==1.0.6 

python wgp.py --i2v-14B --server-name "0.0.0.0" --server-port 7860  --share

# or for Sage Attention 2 (40% faster, sorry only manual compilation for the moment)
git pull https://github.com/thu-ml/SageAttention
cd sageattention 
pip install -e .

# 3.2 optional Flash attention support (easy to install on Linux but much harder on Windows)
python -m pip install flash-attn==2.7.2.post1

```

Note pytorch *sdpa attention* is available by default. It is worth installing *Sage attention* (albout not as simple as it sounds) because it offers a 30% speed boost over *sdpa attention* at a small quality cost.
In order to install Sage, you will need to install also Triton. If Triton is installed you can turn on *Pytorch Compilation* which will give you an additional 20% speed boost and reduced VRAM consumption.

### Ready to use python wheels for Windows users
I provide here links to simplify the installation for Windows users with Python 3.10 / Pytorch 2.51 / Cuda 12.4. I won't be able to provide support neither guarantee they do what they should do.
- Triton attention (needed for *pytorch compilation* and *Sage attention*)
```
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl # triton for pytorch 2.6.0
```

- Sage attention
```
pip install https://github.com/deepbeepmeep/SageAttention/raw/refs/heads/main/releases/sageattention-2.1.0-cp310-cp310-win_amd64.whl # for pytorch 2.6.0 (experimental, if it works, otherwise you you will need to install and compile manually, see above) 
 
```

## Run the application

### Run a Gradio Server on port 7860 (recommended)

To run the text to video generator (in Low VRAM mode): 
```bash
python gradio_server.py
#or
python gradio_server.py --t2v

```

To run the image to video generator (in Low VRAM mode): 
```bash
python gradio_server.py --i2v
```

Within the application you can configure which video generator will be launched without specifying a command line switch.

To run the application while loading entirely the diffusion model in VRAM (slightly faster but requires 24 GB of VRAM for a 8 bits quantized 14B model )
```bash
python gradio_server.py --profile 3
```
Please note that diffusion model of Wan2.1GP is extremely VRAM optimized and this will greatly benefit low VRAM systems since the diffusion / denoising step is the longest part of the generation process. However, the VAE encoder (at the beginning of a image 2 video process) and the VAE decoder (at the end of any video process) is still VRAM hungry after optimization and it will require temporarly 22 GB of VRAM for a 720p generation and 12 GB of VRAM for a 480p generation. Therefore if you have less than these numbers, you may experience slow downs at the beginning and at the end of the generation process due to pytorch VRAM offloading.


### Loras support

-- Ready to be used but theoretical as no lora for Wan have been released as of today. ---

Every lora stored in the subfoler 'loras' will be automatically loaded. You will be then able to activate / desactive any of them when running the application.

For each activated Lora, you may specify a *multiplier* that is one float number that corresponds to its weight (default is 1.0), alternatively you may specify a list of floats multipliers separated by a "," that gives the evolution of this Lora's multiplier over the steps. For instance let's assume there are 30 denoising steps and the multiplier is *0.9,0.8,0.7* then for the steps ranges 0-9, 10-19 and 20-29 the Lora multiplier will be respectively 0.9, 0.8 and 0.7.

You can edit, save or delete Loras presets (combinations of loras with their corresponding multipliers) directly from the gradio interface. Each preset, is a file with ".lset" extension stored in the loras directory and can be shared with other users

Then you can pre activate loras corresponding to a preset when launching the gradio server:
```bash
python gradio_server.py --lora-preset  mylorapreset.lset # where 'mylorapreset.lset' is a preset stored in the 'loras' folder
```

You will find prebuilt Loras on https://civitai.com/ or you will be able to build them with tools such as kohya or onetrainer.


### Command line parameters for Gradio Server
--i2v : launch the image to video generator\
--t2v : launch the text to video generator\
--quantize-transformer bool: (default True) : enable / disable on the fly transformer quantization\
--lora-dir path : Path of directory that contains Loras in diffusers / safetensor format\
--lora-preset preset : name of preset gile (without the extension) to preload
--verbose level : default (1) : level of information between 0 and 2\
--server-port portno : default (7860) : Gradio port no\
--server-name name : default (0.0.0.0) : Gradio server name\
--open-browser : open automatically Browser when launching Gradio Server\
--compile : turn on pytorch compilation\
--attention mode: force attention mode among, sdpa, flash, sage, sage2\
--profile no : default (4) : no of profile between 1 and 5

### Profiles (for power users only)
You can choose between 5 profiles, but two are really relevant here :
- LowRAM_HighVRAM  (3): loads entirely the model in VRAM, slighty faster, but less VRAM
- LowRAM_LowVRAM  (4): load only the part of the models that is needed, low VRAM and low RAM requirement but slightly slower


### Other Models for the GPU Poor

- HuanyuanVideoGP: https://github.com/deepbeepmeep/HunyuanVideoGP :\
One of the best open source Text to Video generator

- Hunyuan3D-2GP: https://github.com/deepbeepmeep/Hunyuan3D-2GP :\
A great image to 3D and text to 3D tool by the Tencent team. Thanks to mmgp it can run with less than 6 GB of VRAM

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP :\
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP :\
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).

- OminiControlGP: https://github.com/deepbeepmeep/OminiControlGP :\
A Flux derived application very powerful that can be used to transfer an object of your choice in a prompted scene. With mmgp you can run it with only 6 GB of VRAM.

- YuE GP: https://github.com/deepbeepmeep/YuEGP :\
A great song generator (instruments + singer's voice) based on prompted Lyrics and a genre description. Thanks to mmgp you can run it with less than 10 GB of VRAM without waiting forever.



# StableDiffusionUi

Simple Stable Diffusion UI with as little dependencies as possible. And with a lot of sliders to findout which params and values influence output image content and how. sadly only 2 shedulers so far but only relevant third one is euler_a anyway. will add soon

install python 3.10.7
install cudatoolkit 11.7 . I  got twice the speed on 3080 compared to 11.6 11 it/s compared to 6 but only on windows linux was half of the speed . win drivers are 516 linux is stuck on 515 perhaps thats why. also on linux I get constant out of memory failures in 32bit float on windows not. what is strange xorg was taking just 177m vram and windows 210m but windows cura did not complained in 32bit and got no out of memory on 3080 so windows is much better testbed right now which is weird.
dint forget to run 
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 on windows or linux if you get out of memory errors

also I disabled nsfw since it was taking 2g of precious videoram that is why this can now run 32bit

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 <br>

install latest diffusers 0.4 from github directly so we get image 2image where pip install well get you only 0.3 without it

pip install https://github.com/huggingface/diffusers/archive/refs/heads/main.zip


pip install --upgrade  transformers scipy ftfy gradio

run via : 

gradio StableDiffusionUi.py 

it will start local webserver ans show on which port

open browser http://localhost:shown_port to see html based gui

upscale results manually with qualityscaller ai upscaller for example install qualityscalle from github
realsr_jpeg results there are superior to blury bsrgan


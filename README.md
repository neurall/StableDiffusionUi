# StableDiffusionUi

Simple Stable Diffusion UI with a lot of sliders to findout which params and values influence output image content and how.

install python 3.10.7
install cudatoolkit 11.7 . I  got twice the speed on 3080 compared to 11.6 11 it/s compared to 6 bot only on windows linux was half of the speed . win drivers are 516 linux is stuck on 515 perhaps thats why

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


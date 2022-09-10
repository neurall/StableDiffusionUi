
import torch,os,time,requests,sys,functools,IPython
import gradio as gr
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import LMSDiscreteScheduler
from PIL import Image
from io import BytesIO

torch.cuda.empty_cache(); torch.cuda.synchronize()

prompt = "futuristic vines across scifi bamboo stems cityscape flying cars bladerunner cyberpunk hdr octane lush green dark detailed bautifull romantic"#"bamboo stem node cut skyscrapper sci-fi cityscape futuristic flying-cars bladerunner cyberpunk hdr octane detailed bautifull"
device = "cuda"; model_id = "CompVis/stable-diffusion-v1-4"; 

scheduler=False; low=False; useimg=False; pipe=None; init_image=None

def newpipe(pr,steps,guidance,seed,strength,s1,s2,s3,sch,lr,uim,hlf,inm):
    global pipe, scheduler, useimg, init_image, low
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    pa = {}
    if lr:
        pa["revision"]="fp16"; pa["torch_dtype"] = torch.float16
    if sch:
        pa["scheduler"]=LMSDiscreteScheduler(beta_start=s1, beta_end=s2, beta_schedule="scaled_linear", num_train_timesteps=s3)
    if uim:
        init_image = Image.open(inm).convert("RGB"); init_image = init_image.resize((512, 512))    
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,use_auth_token=True,**pa)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id,use_auth_token=True,**pa)
    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy
    pipe = pipe.to(device)
    low=lr; scheduler = sch; useimg=uim

inputs = [
    gr.Textbox(prompt),
    gr.Slider(1, 250,step=1, label='iters', value=15),
    gr.Slider(0, 10,step=0.5, label='guid', value=7.5),
    gr.Slider(0, 0xffffffff,step=1, label='seed', value=0),
    gr.Slider(0, 1,step=0.001, label='blend', value=0.45),
    gr.Slider(0.0001, 0.012,step=0.0001, label='sch from', value=0.00085),
    gr.Slider(0.001, 0.012,step=0.0001,  label='sch to', value=0.012),
    gr.Slider(0, 1200,step=100, label='sch iters', value=1000),
    "checkbox","checkbox","checkbox","checkbox", gr.Image(type="filepath", value="a.jpg")
]

def work(pr,steps,guidance,seed,strength,s1,s2,s3,sch,lr,uim,hlf,inm):  #url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Flh6.googleusercontent.com%2Fproxy%2FQBZTSycU3BmNf_YnyU4vm7Ammqx-aKnQcXSr_mD5xn28buzYMdg4G4NZOoxF8-hhW-a_HenRvkeiOnwnRw4Ll0TYjcs9HehMcoFWWsCpqA5BKAQCOUeJ5yd2eXnIlcxd_F6n7dugodr4GuwtI7aNtIqETNci0EFPeyc6sgVUx4f_1wHb_RwEXRGTxbc3L5jqYbde-ArPVo4_HRuJPBhmq-hJ86M4%3Dw1200-h630-p-k-no-nu&f=1&nofb=1"#response = requests.get(url)#BytesIO(response.content)
    if s1>s2:
        s2=s1+0.006
    
    if (bool(sch) != bool(scheduler)) or (bool(lr) != bool(low)) or (bool(uim) != bool(useimg)) or not pipe:
        newpipe(**locals())

    dr = "c:\\i"#\\"+pr.replace(' ','_'); 

    #if not os.path.exists(dr):
    #    os.makedirs(dr)

    fname = "c:\i\\"+pr.replace(' ','_')+str(seed)+'_'+str(guidance)+'_'+str(steps)+".jpg"

    pa={"width":512,"height":512}
    if uim:
        pa={"init_image":init_image, "strength":strength}
    if hlf:
        pa["width"]=pa["height"]=256
    with autocast("cuda"):
        image = pipe(pr, guidance_scale=guidance, num_inference_steps=steps ,generator=torch.Generator("cuda").manual_seed(seed),**pa)["sample"][0]  

    image.save(fname)
    
    while not os.path.exists(fname):
        time.sleep(0.1)

    os.startfile(fname)
    return image

demo = gr.Interface(fn=work, inputs=inputs, live=True, outputs=gr.Image(source="canvas",interactive=True))
demo.launch()



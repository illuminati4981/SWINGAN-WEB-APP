import argparse
import os
import pickle
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import legacy
import dnnlib
from torch_utils import misc
from CustomSwin import CustomSwin
from training.networks import Generator


def init_models(model_path):
    swin = CustomSwin().requires_grad_(False)
    
    G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    G_kwargs.synthesis_kwargs.channel_base = int(0.5 * 32768)
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = 2
    G_kwargs.synthesis_kwargs.num_fp16_res = 4 # enable mixed-precision training
    G_kwargs.synthesis_kwargs.conv_clamp =  256 # clamp activations to avoid float16 overflow\
    common_kwargs = dict(
        c_dim=0, 
        img_resolution=256,
        img_channels=3,
    )

    generator = (
        dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs)
        .requires_grad_(False)
    ) 
    with dnnlib.util.open_url(model_path) as f:
        resume_data = legacy.load_network_pkl(f)
    for name, module in [("swin", swin), ("G", generator)]:
        misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    return swin, generator

def perform_inference(normalised_input_tensor, swin, generator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swin.to(device)
    generator.to(device)
    swin.eval() 
    generator.eval()
    normalised_input_tensor = normalised_input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        x, size128_output, size64_output, size32_output, size16_output, size8_output, size4_output = swin(normalised_input_tensor)
        noises = [size128_output, size64_output, size32_output, size16_output, size8_output, size4_output]
        output = generator(x, 0, 0, noises)
    return output

def process_images(input_image, swin, generator):
    normalised_input_image = ToTensor()(input_image)*2 - 1
    generated_image = perform_inference(normalised_input_image, swin, generator)
    output_image = ToPILImage()(((generated_image.squeeze(0)+1)*127.5).cpu())
    return output_image

def restore_image(model_path, input_image):
    swin, generator = init_models(model_path)
    return process_images(input_image, swin, generator)

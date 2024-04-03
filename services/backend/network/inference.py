import argparse
import os
import pickle
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms import PILToTensor
import legacy
from torch_utils import misc
from CustomSwin import CustomSwin
from training.networks import Generator
import numpy as np
from PIL import Image
import dnnlib

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
    normalised_input_tensor = normalised_input_tensor.to(device)
    
    with torch.no_grad():
        x, size128_output, size64_output, size32_output, size16_output, size8_output, size4_output = swin(normalised_input_tensor)
        noises = [size128_output, size64_output, size32_output, size16_output, size8_output, size4_output]
        gen_img = generator.mapping(x, 0)
        output = generator.synthesis(gen_img, noises)
    return output

    
def process_images(input_image, swin, generator):
    normalised_input_image = PILToTensor()(input_image).to(torch.float32).unsqueeze(0)
    normalised_input_image = normalised_input_image / 127.5 - 1

    generated_image = perform_inference(normalised_input_image, swin, generator)
    return generated_image


def restore_image(model_path, input_image, output_path):
    swin, generator = init_models(model_path)
    img = process_images(input_image, swin, generator)
    if output_path:
        save_images(img, output_path)
    return img


def save_images(img, output_path):
    img = img.to('cpu').numpy()
    output_image = np.asarray(img, dtype=np.float32)
    output_image = ((output_image + 1) * 127.5).squeeze()
    output_image = np.rint(output_image).clip(0, 255).astype(np.uint8)
    output_image = output_image.transpose(1, 2, 0)
    Image.fromarray(output_image, "RGB").save(output_path)
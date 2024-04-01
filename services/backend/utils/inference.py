import argparse
import os
import pickle
import torch
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import ToPILImage

def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model

def perform_inference(model, input_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 
    input_tensor = input_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    return output

def process_images(model, input_image):
    input_image = ToTensor()(input_image)/127.5 - 1
    generated_image = perform_inference(model, input_image)
    output_image = ToPILImage()(((generated_image.squeeze(0)+1)*127.5).cpu())
    return output_image

def restore_image(model_path, input_image):

    model = load_model(model_path)
    
    return process_images(model, input_image)

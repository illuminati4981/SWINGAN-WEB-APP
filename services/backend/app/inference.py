import argparse
import os
import pickle
import torch
from torchvision.io import read_image
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

    output_image = ToPILImage()(((output.squeeze(0)+1)*127.5).cpu())
    return output_image

def process_images(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            input_image = read_image(input_path)/127.5 - 1
            generated_image = perform_inference(model, input_image)
            generated_image.save(output_path)
            
            print(f"Inference completed for {filename}. Output saved to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Directory path of the model checkpoint")
    parser.add_argument("input_dir", type=str, help="Directory path containing the input images")
    parser.add_argument("output_dir", type=str, help="Directory path to save the output images")

    args = parser.parse_args()

    model = load_model(args.model_dir)
    
    process_images(model, args.input_dir, args.output_dir)

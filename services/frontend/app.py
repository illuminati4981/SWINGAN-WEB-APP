import gradio as gr
import requests
from PIL import Image
from gradio_imageslider import ImageSlider
import io
from os import path

# ----------------------------------- setup ---------------------------------- #
# before_image = Image.open("sample/React-icon.png")
# after_image = Image.open("sample/Vue.js_Logo.png")
swin_img = Image.open("sample/SwinTransformer.png")
pipeline_img = Image.open("sample/pipeline.png")
deg_mod_img = Image.open("sample/deg_mod.png")
timeout = 60


# -------------------------------- to backend -------------------------------- #
endpoint_url = "http://localhost:5000/generate"
def restore_handler(filepath, checkpoint):    
    if filepath is None:   
        gr.Warning("There is no image.")
        return None

    filename = path.split(filepath)[-1]
    fileformat = filename.split('.')[-1]

    if fileformat.lower() not in ['png', 'jpeg', 'jpg']:
        gr.Warning("The file format should be .png .jpeg .jpg \t Your file format: " + fileformat)
        return None

    image  = Image.open(filepath)
    if image.size != (256, 256):
        gr.Warning("The image resolution should be 256 x 256. \t Your image has resolution : ", image.size)
        return None
    

    image_io = io.BytesIO()
    image.save(image_io, format='PNG')
    image_io.seek(0)


    params = {"checkpoint" : checkpoint}
    files = {"input_image": (filename, image_io, "image/png")}

    try:
        response = requests.post(endpoint_url, files=files, params=params, timeout=timeout)
        if response.ok:
            gen_img = Image.open(io.BytesIO(response.content))
            return gen_img
        else:
            gr.Warning("Bad response: status : ", response.status_code)
            return None
    except Exception as e:
        gr.Warning("Request error")
        # print(e.with_traceback())
        return None

# --------------------------------- interface -------------------------------- #
MARKDOWN1 = \
"""
# SwinGAN - Blind Image Restoration with Swin Transformer + StyleGAN2-ADA

<br>

### Generative AI on Visual Content Creation (CQF1)
> Advised by Prof. Qifeng CHEN <br>
> CHONG Ho Yuen, LI Yiu Ting, KWAN Kam To Christopher <br>

---
"""

MARKDOWN2 = \
"""
## Abstract
> This project aims to develop deep neural network that can perform blind image restoration tasks under various real-world scenario, the real-world scenarios often involve a combination of various degradations such as blur, resize, noise, JPEG artifacts, and low resolution. In this project, we propose a new deep neural network model Swin-GAN, which integrate Swin V2 Transformer as a feature extractor into StyleGAN2-ADA generative adversarial network (GAN). In order to better generate the restored image, we also integrate U-Net structure on the Swin-GAN deep neural network. We hope the work could provide a robust and efficient solution that addresses the current challenges in the field of blind image restoration.

<br>

## Our focus
> In this project, we propose a new deep neural network model SwinGAN combined with a degradation layer in the training pipeline to perform general image restoration under various degradations with great balance of fidelity and quality. It is hoped that our work could provide a robust and efficient solution that addresses the current challenges in the field of blind image restoration.

---
"""

MARKDOWN3 = \
"""
### Input specification
* Input image should have width 256 and height 256.
### Steps
1. Click to upload the image from local directory or drag-drop to the box
2. Choose the checkpoint from the drop-down list
3. Click on the sumbit button to sumbit; or Click Clear button to clear the inputted image.

"""


block_css = \
"""
footer{display:none !important}
h1{justify-content: center;}

div:has(#img-input, #img-output) {
    align-items: center;
}


#img-output {
    justify-self: center;
}

#img-slider img{
  object-fit: contain;
}
"""

interface_css = \
"""
.image-container {
    display: flex;
    justify-content: center;
    justify-self: center;
    align-items: center;
}
"""

block = gr.Blocks(
    title="SwinGAN",
    css=block_css,
    theme=gr.themes.Monochrome(),
).queue()

# TODO add the checkpoint filename into here
checkpoint_choice = [
    "network-snapshot-019201.pkl",
    "network-snapshot-008750.pkl"
]

inputs = [
    gr.Image(type='filepath', sources=["upload"], label="Input Image", elem_id="img-input", interactive=True, height=192, width=192),
    gr.Dropdown(choices=checkpoint_choice, value=checkpoint_choice[0], label="Checkpoint")
]

outputs = [
    gr.Image(label="Generated Image", elem_id="img-output", height=192, width=192, show_download_button=True, type="filepath")
]

gallery_picture = [f"gallery/result_16_{i}_gt.png" for i in range(10)] \
+ [f"gallery/result_16_{i}_deg.png" for i in range(10)] \
+ [f"gallery/result_16_{i}_restored.png" for i in range(10)]

with block as demo:
    with gr.Column():
        gr.Markdown(MARKDOWN1)
        gr.Gallery(
            value=gallery_picture,
            rows=[3],
            columns=[10],
            object_fit="contain",
            show_label=False,
            allow_preview=False,
            height="max",
            container=False
        )
        # ImageSlider(value=(before_image, after_image), label="Before & After", show_download_button=True, elem_id="img-slider")
        gr.Markdown(MARKDOWN2)
        gr.Markdown("## **SWINGAN**")
        with gr.Tab("Restored image with SwinGAN"):
            with gr.Tab("Restore your image"):
                gr.Markdown(MARKDOWN3)
                gr.Interface(
                    fn=restore_handler,
                    inputs=inputs,
                    outputs=outputs,
                    allow_flagging="never",
                    css=interface_css
                )
        with gr.Tab("Our restoration network"):
            # gr.Markdown("Under construction")
            gr.Image(value=swin_img, label="Architecture of Image restoration network")
            gr.Image(value=pipeline_img, label="Training pipeline of Image restoration network")
        with gr.Tab("Our degradation module"):
            # gr.Markdown("Under construction")
            gr.Image(value=deg_mod_img, label="Detailed Diagram of second-order degrdation module")

if __name__ == "__main__":
    demo.launch(debug=True)
import gradio as gr
import requests
from PIL import Image
from gradio_imageslider import ImageSlider
import io
from os import path

# ----------------------------------- setup ---------------------------------- #
before_image = Image.open("sample/React-icon.png")
after_image = Image.open("sample/Vue.js_Logo.png")

# -------------------------------- to backend -------------------------------- #
endpoint_url = "http://localhost:5000/generate"
def restore_handler(filepath):
    if filepath is None:   
        gr.Warning("There is no image.")
        return None

    filename = path.split(filepath)[-1]
    fileformat = filename.split('.')[-1]

    if fileformat.lower() not in ['png', 'jpeg', 'jpg']:
        gr.Warning("The file format should be .png .jpeg .jpg \t Your file format: " + fileformat)
        return None

    image  = Image.open(filepath)
    image_io = io.BytesIO()
    image.save(image_io, format='PNG')
    image_io.seek(0)

    files = {"input_image": (filename, image_io, "image/png")}
    try:
        response = requests.post(endpoint_url, files=files)
        if response.ok:
            gen_img = Image.open(io.BytesIO(response.content))
            return gen_img
        else:
            gr.Warning("Wrong response")
            return None
    except Exception as e:
        gr.Warning("Request error")
        print(e.with_traceback())
        return None

# --------------------------------- interface -------------------------------- #
MARKDOWN1 = \
"""
# SwinGAN - Image Restoration
#### Generative AI on Visual Content Creation (CQF1)
> Advised by Prof. Qifeng CHEN <br>
> CHONG Ho Yuen, LI Yiu Ting, KWAN Kam To Christopher <br>

---
"""

MARKDOWN2 = \
"""
## Background
> Blind image restoration is the process of recovering an image from its degraded observation without any prior knowledge about the degradation.\
Degradations can occur during image acquisition and transmission due to factors like noise, blurring, compression artifacts etc.

## Our foucs
> Our blind image restoration aims to estimate the latent clean image from a single corrupted input image.

---
"""
block_css = \
"""
footer{display:none !important}
h1{justify-content: center;}


#img-input, #img_output{
    align-items: center;
    justify-self: center;
    justify-content: center;
}

#img-slider {
  display: flex;
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

input_image = gr.Image(type='filepath', sources=["upload"], label="Input Image", elem_id="img-input", interactive=True)
generated_image = gr.Image(label="Generated Image", elem_id="img-output")

with block as demo:
    with gr.Column():
        gr.Markdown(MARKDOWN1)
        ImageSlider(value=(before_image, after_image), label="Before & After", show_download_button=True, elem_id="img-slider")
        gr.Markdown(MARKDOWN2)
        gr.Markdown("## Restored image with SwinGAN")
        with gr.Tab("Restore your image"):
            gr.Interface(
                fn=restore_handler,
                inputs=input_image,
                outputs=generated_image,
                allow_flagging="never",
                css=interface_css
            )
        # with gr.Tab("Restored your image under degradation"):
        #     gr.Markdown("Under construction")

if __name__ == "__main__":
    demo.launch(debug=True)
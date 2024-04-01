from inference import restore_image
from PIL import Image


input_path = './0000.png'
model_path = './network-snapshot-019201.pkl'
output_path = './output.png'

input_image = Image.open(input_path)
restored_image = restore_image(model_path, input_image)
restored_image.save(output_path)

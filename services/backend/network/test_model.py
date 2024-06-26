from inference import restore_image
from PIL import Image

input_path = '../dataset/00055.png'
model_path = '../checkpoint/network-snapshot-019201.pkl'
output_path = '../output/result.png'

input_image = Image.open(input_path)
restored_image = restore_image(model_path, input_image, output_path)
# restored_image.save(output_path)

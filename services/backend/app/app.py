from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import os
import subprocess
from datetime import datetime
import zipfile
from inference import restore_image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/upload"
app.config["OUT_FOLDER"] = "static/out"
app.config["ZIP_FOLDER"] = "static/zip"
app.config["MODEL_PATH"] = "static/model"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

def generate_image(model_path, input_image):
    generated_image = restore_image(model_path, input_image)
    return generated_image

def zip_image(image, filename):
    img_buffer = io.BytesIO()
    image.save(img_buffer, 'PNG')
    img_buffer.seek(0)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr(filename, img_buffer.read())
        zip_buffer.seek(0)

    with open(filename, 'wb') as f:
        f.write(zip_buffer.getvalue())

# Define the route for image generation
@app.route("/generate", methods=["POST"])
def generate():
    if "input_image" not in request.files:
        return jsonify({"error": "No input image found."}), 400

    file = request.files["input_image"]
    if file and allowed_file(file.filename):
        input_image = Image.open(file)
        
        utc_timestamp = datetime.utcnow()

        timestamp_str = utc_timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        input_image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file.filename}_in_{timestamp_str}.png")
        input_image.save(input_image_path)

        input_zip_path = os.path.join(app.config["ZIP_FOLDER"], f"{file.filename}_zip_{timestamp_str}.zip")
        zip_image(input_image, input_zip_path)

        generated_image = generate_image(app.config["MODEL_PATH"], input_image)

        generated_image_path = os.path.join(app.config["OUT_FOLDER"], f"{file.filename}_out_{timestamp_str}.png")
        generated_image.save(generated_image_path)

        generated_image_io = io.BytesIO()
        generated_image.save(generated_image_io, format='PNG')
        generated_image_io.seek(0)

        return send_file(generated_image_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')

    return jsonify({"error": "Invalid file."}), 400

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

if __name__ == "__main__":
    app.run(debug=True)
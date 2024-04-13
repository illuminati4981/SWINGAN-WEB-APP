from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import os
from datetime import datetime
import zipfile
# from inference import restore_image
restore_image = lambda *x : None

app = Flask(__name__)

# ---------------------------------- config ---------------------------------- #
app.config["RECORDS_DIR"] = "../static/records"
app.config["MODEL_PATH"] = "../checkpoint/"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# --------------------------------- function --------------------------------- #

def get_filename(filename) -> str:
    return filename.rsplit(".", 1)[0] if "." in filename else filename
    
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

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

def generate_image(model_path, input_image, filename):
    # ---------------------------- make save directory --------------------------- #
    timestamp_str = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    record_dir = os.path.join(app.config["RECORDS_DIR"], f"record_{timestamp_str}")
    assert not os.path.exists(record_dir), f"{record_dir} should not be existed."
    os.mkdir(record_dir)

    # ------------------------- input, zip, out filepath ------------------------- #
    input_image_path = os.path.join(record_dir, f"{filename}_in.png")
    input_zip_path = os.path.join(record_dir, f"{filename}_compressed.zip")
    out_filename = f"{filename}_out.png" 
    generated_image_path = os.path.join(record_dir, out_filename)

    # --------------------------- input, zip, out save --------------------------- #
    input_image.save(input_image_path, format='PNG')
    zip_image(input_image, input_zip_path)
    # generated_image.save(generated_image_path, format='PNG')

    restore_image(model_path, input_image, output_path=generated_image_path)
    generated_image = Image.open(generated_image_path)
    
    return generated_image

# ----------------------------------- Route ---------------------------------- #

@app.route("/generate", methods=["GET", "POST"])
def generate():
    if "input_image" not in request.files:
        return jsonify({"error": "No input image found."}), 400
    
    file = request.files["input_image"]
    if file is None or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file."}), 400

    checkpoint = request.args.get("checkpoint")
    model_path = os.path.join(app.config["MODEL_PATH"], checkpoint)
    input_image = Image.open(file)
    # ------------------------------ Generate image ------------------------------ #
    filename = get_filename(file.filename)
    generated_image = generate_image(model_path, input_image, filename)
    # --------------------- Generated PIL image to io stream --------------------- #
    generated_image_io = io.BytesIO()
    generated_image.save(generated_image_io, format='PNG')
    generated_image_io.seek(0)
    out_filename = f"{filename}_out.png" 
    # --------------------------------- send file -------------------------------- #
    return send_file(generated_image_io, mimetype='image/png', as_attachment=True, download_name=out_filename)

if __name__ == "__main__":
    app.run(debug=True)
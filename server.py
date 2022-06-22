# _*_ coding: utf-8 _*_
import os
from flask import Flask, render_template, request, redirect
from flask import url_for, flash, redirect
from werkzeug.utils import secure_filename
import sys
from white_box_cartoonizer.cartoonize import WB_Cartoonize
from PIL import Image
import numpy as np
import io
import yaml
import flask
import cv2
import uuid
import traceback
with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)


sys.path.insert(0, './white_box_cartoonizer/')

if not opts['run_local']:
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        from gcloud_utils import upload_blob, generate_signed_url, delete_blob, download_video
    else:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set in environment variables")

app = Flask(__name__)
app.secret_key = 'random string'

checkpoint_dir = "model/generator_Hayao_weight"
test_dir = "static/image"
style_name = "Hayao"

app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_images'
app.config['OPTS'] = opts

wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):
    """Convert bytes to numpy array

    Args:
        img_bytes (bytes): Image bytes read from flask.

    Returns:
        [numpy array]: Image numpy array
    """
	
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode=="RGBA":
        image = Image.new("RGB", pil_image.size, (255,255,255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    
    image = np.array(image)
    
    return image

# @app.route("/", methods = ['GET', 'POST'])
# def index():
# 	return render_template('index.html')

# @app.route("/cartoon")
# def cartoon():
# 	return render_template('photo.html')

@app.route('/')
@app.route('/cartoonize', methods=["POST", "GET"])
def cartoonize():
    opts = app.config['OPTS']
    if flask.request.method == 'POST':
        try:
            if flask.request.files.get('image'):
                img = flask.request.files["image"].read()
                
                ## Read Image and convert to PIL (RGB) if RGBA convert appropriately
                image = convert_bytes_to_image(img)

                img_name = str(uuid.uuid4())
                
                cartoon_image = wb_cartoonizer.infer(image)
                
                cartoonized_img_name = os.path.join(app.config['CARTOONIZED_FOLDER'], img_name + ".jpg")
                cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
                
                if not opts["run_local"]:
                    # Upload to bucket
                    output_uri = upload_blob("cartoonized_images", cartoonized_img_name, img_name + ".jpg", content_type='image/jpg')

                    # Delete locally stored cartoonized image
                    os.system("rm " + cartoonized_img_name)
                    cartoonized_img_name = generate_signed_url(output_uri)
                    

                return render_template("result.html", cartoonized_image=cartoonized_img_name)
        
        except Exception:
            print(traceback.print_exc())
            flash("Our server hiccuped :/ Please upload another file! :)")
            return render_template("index.html")
    else:
        return render_template("index.html")


if __name__ == '__main__':
	app.run(debug = True)
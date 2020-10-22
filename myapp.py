from flask import Flask,render_template,url_for,redirect,request,flash
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from skimage import transform
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode

UPLOAD_FOLDER = 'C:\\Users\\vybha\\Desktop\\College stuff\\sem 5\\Ai in mfg\\Dataset\\NEU-DET\\uploads'
static_folder='C:\\Users\\vybha\\Desktop\\College stuff\\sem 5\\Ai in mfg\\Dataset\\NEU-DET\\static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY']='mysecretkey'

def preprocessimg(img):
    im=np.asarray(img)
    im=transform.resize(im,(224,224))
    im=np.expand_dims(im,axis=0)
    return im

def make_prediction(img,model):
    pred=model.predict_proba(img)
    p1=np.argmax(pred)
    mapper={0:'Crazing',1:'Inclusion',2:'Patches',3:'Pitted Surface',4:'Rolled in Scale',5:'Scratches'}
    p=mapper[p1]
    return pred,p,p1


@app.route('/')
def main_page():
    return render_template('home.html')


@app.route('/',methods=['POST'])
def prediction():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        flash('Image successfully uploaded and displayed')
        model=load_model('C:\\Users\\vybha\\Desktop\\College stuff\\sem 5\\Ai in mfg\\Dataset\\NEU-DET\\static\\Finalmodel.h5')
        image = plt.imread(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        my_image_re = preprocessimg(image)
        p,prediction,p1=make_prediction(my_image_re,model)
        x=p[0,p1]
        predictions={'class':prediction,'probability':x}
        original_img = Image.open(file)
        byteIO = BytesIO()
        original_img.save(byteIO,format=original_img.format)
        byteArr = byteIO.getvalue()
        encoded = b64encode(byteArr)
        return render_template('prediction.html', predictions=predictions,image=encoded.decode('ascii'))


if __name__=='__main__':
	app.run(debug=True)

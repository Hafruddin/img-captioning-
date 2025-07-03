from flask import Flask, render_template, request
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load models and mappings
model = load_model('caption_model.h5')  # Pre-trained image captioning model
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
max_length = 34  # adjust to your model's setting

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = resnet.predict(x)
    return feature

def generate_caption(photo):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant')
        yhat = model.predict([photo, np.array([sequence])], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()

@app.route("/", methods=["GET", "POST"])
def index():
    caption = ""
    if request.method == "POST":
        img = request.files["image"]
        path = "static/uploaded_image.jpg"
        img.save(path)
        photo = extract_features(path)
        caption = generate_caption(photo)
    return render_template("index.html", caption=caption)

if __name__ == "__main__":
    app.run(debug=True)
import streamlit as st 
from PIL import Image
import ser_prediction
import gan_prediction
from tensorflow.keras.models import load_model

# @st.cache()
def get_ser_model():
	weights = '/home/hs/Desktop/Projects/speech_emotion_recognition/emotion-recognition/saved_models/Emotion_Voice_Detection_Model.h5'
	model = load_model(weights)
	return model

# @st.cache()
def get_cgan_model():
	weights = '/home/hs/Desktop/Projects/speech_emotion_recognition/cgan_epoch400.h5'
	model = load_model(weights)
	return model

st.markdown("<h1 style='text-align: center; color: red;'>Speech Emotion Recognition</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: Black;'>Select the audio file to predict emotion</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose audio file...", type="wav")
if uploaded_file is not None:
	model = get_ser_model()
	emotion = ser_prediction.predict(model, uploaded_file)
	emotion = emotion[0].upper() + emotion[1:]
	st.write(' ')
	st.markdown("<h4 style='text-align: center; color: black;'>Predicted Emotion</h1>".format(emotion), unsafe_allow_html=True)
	st.markdown("<h2 style='text-align: center; color: green;'>{}</h1>".format(emotion), unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: red;'>Conditional GAN</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: Black;'>Generating a face with predicted emotion</h1>", unsafe_allow_html=True)

model = get_cgan_model()

emotion_2_id = {'angry':0,
				'disgusted':1,
				'fearful':2,
				'happy':3,
				'sad':4,
				'surprised':5,
				'neutral':6}

def generate_img():
	gan_prediction.predict(model, emotion_2_id[emotion.lower()])
	image = Image.open('./generated_img.jpg')
	# image=image.resize((144,144))
	st.image(image, width=200)

generate_btn = st.button("Generate Image")
if generate_btn:
	generate_img()
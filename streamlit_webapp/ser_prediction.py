# import pyaudio
import wave
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from keras.models import load_model


def predict(model, file):
	X, sample_rate = librosa.load(file, res_type='kaiser_fast',duration=3,sr=22050*2,offset=0.5)
	sample_rate = np.array(sample_rate)

	#opt = rmsprop(lr=0.0001, decay=1e-6)
	#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	labelled_class_path = '/home/hs/Desktop/Projects/speech_emotion_recognition/emotion-recognition/label_classes.npy'
	lb = LabelEncoder()
	lb.classes_ = np.load(labelled_class_path, allow_pickle=True)
	features = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
	features_df= pd.DataFrame(data=features)
	features_stacked = features_df.stack().to_frame().T
	features_expanded= np.expand_dims(features_stacked, axis=2)
	
	predictions = model.predict(features_expanded, batch_size=1, verbose=1)
	predictions_mod = predictions.argmax(axis=1)
	preds_flat = predictions_mod.astype(int).flatten()
	predictions_array = (lb.inverse_transform((preds_flat)))
	predictions_array
	return predictions_array[0]

if __name__=="__main__":
	weights = '/home/hs/Desktop/Projects/speech_emotion_recognition/emotion-recognition/saved_models/Emotion_Voice_Detection_Model.h5'
	model = load_model(weights)
	file_path='/home/hs/Desktop/Projects/speech_emotion_recognition/emotion-recognition/Audio_Speech_Actors_01-24/Actor_09/03-01-03-02-01-02-09.wav'
	emotion=predict(model, file_path)
	print(emotion)
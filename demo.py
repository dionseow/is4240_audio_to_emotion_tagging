# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:31:43 2017

@author: dion
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import seaborn as sns
import numpy as np
import pickle
import pydub
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioBasicIO

##
#To Visualize the audio file
#==============================================================================
# import matplotlib.pyplot as plt
# from scipy.io import wavfile as wav
# audio_file = r"C:\Users\dion\Desktop\Downloads\4240\demo\s1s1an1.mp3"
# to_predict = pydub.AudioSegment.from_mp3(audio_file)
# to_predict.export(r"C:\Users\dion\Desktop\Downloads\4240\demo\s1s1an1.wav", format="wav")
# rate, data = wav.read(r"C:\Users\dion\Desktop\Downloads\4240\demo\s1s1an1.wav")
# plt.plot(data)
# plt.show()
#==============================================================================
#Prediction code
#==============================================================================
 ##### Read mp3 and convert to features #####
audio_file = "s1s1an1.mp3"
to_predict = pydub.AudioSegment.from_mp3(audio_file)
[Fs, x] = audioBasicIO.readAudioFile(to_predict);
results = aF.stFeatureExtraction(x, Fs, 0.1*Fs, 0.1*Fs)
results = pd.DataFrame(results)
##### Load normalizer #####
scaler = pickle.load(open("normalize.pkl","rb"))
results_normalized = scaler.transform(results)
results_normalized = pd.DataFrame(results_normalized)
results_feature_select = results_normalized.iloc[:, [0, 6, 9, 20]]
##### Load Model #####
model = pickle.load(open("gb_500.pkl","rb"))
predicted = model.predict(results_feature_select)
 
#==============================================================================

import flask
import test
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import random 
import operator

import math
import numpy as np
from collections import defaultdict

from flask import request, jsonify
import requests
from pydub import AudioSegment

import random
app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    if 'music_file' not in request.files :
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    
    results= ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    convertMP3ToWav(request.files['music_file'])
    (rate,sig)=wav.read('music_file.wav')
    mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature=(mean_matrix,covariance,0)

    pred= test.nearestClass(test.getNeighbors(test.dataset ,feature , 5))
    callNodeApi(results[pred])
    return jsonify({"type" : results[pred]}),200

song_names = {'song_one', 'song_two', 'song_third', 'song_fourth', 'song_fifth', 'song_sixth'}
singer_names = {'singer_one', 'singer_two', 'singer_third', 'singer_fourth', 'singer_fifth', 'singer_sixth'}
moods = {'Happiness','Sadness', 'Fear', 'Disgust', 'Anger', 'Surprise'}
occasions = {'occasion_one', 'occasion_two',  'occasion_third', 'occasion_fourth', 'occasion_fifth', 'occasion_sixth'}
years = {2014, 2015, 2016, 2017, 2018, 2019}
albums = {'album_one', 'album_two', 'album_third', 'album_fourth', 'album_fifth', 'album_sixth'}
def callNodeApi(type):
    params = {
    'api_key': '{API_KEY}',
    "song_genre": type,
    "song_name": random.choice(tuple(song_names)),
    "singer_name":random.choice(tuple(singer_names)),
    "mood": random.choice(tuple(moods)),
    "occasion": random.choice(tuple(occasions)),
    "year": random.choice(tuple(years)),
    "album": random.choice(tuple(albums))
    }
    r = requests.post('http://127.0.0.1:3000/api/songs', params)

def convertMP3ToWav(file):
    dst = "music_file.wav"

    # convert mp3 file to wav file 
    sound = AudioSegment.from_mp3(file)
    sound.export(dst, format="wav")

app.run()

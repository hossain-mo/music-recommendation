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

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    if 'music_file' not in request.files :
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    
    results= ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    (rate,sig)=wav.read(request.files['music_file'])
    mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature=(mean_matrix,covariance,0)

    pred= test.nearestClass(test.getNeighbors(test.dataset ,feature , 5))

    return jsonify({"type" : results[pred]}),200

app.run()
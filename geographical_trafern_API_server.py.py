import pandas as pd
import numpy as np

from flask import Flask, request, jsonify

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

app = Flask(__name__)
imns = None
map_porto = None

def init():
    return True

@app.route('/new_training', methods=['POST'])
def new_training():
    input_data = request.get_json()
    config = AttributeDict(input_data)

    



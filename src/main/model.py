from flask import Blueprint
import joblib
import torch
import numpy as np
import os

blue_model = Blueprint('model', __name__)

@blue_model.route('/clothes')
def clothes_model():
    model_path = os.path.join(get_current_path(), '..', 'model', 'clothes_model.pkl')
    clf = joblib.load(model_path)
    sample = torch.FloatTensor(np.array([1, 1, 5, 3, 0, -14, 1]))
    answer = torch.argmax(clf(sample)).detach().numpy()
    return str(answer)

@blue_model.route('/region')
def region_model():
    return "success_region_model";

@blue_model.route('/schedule')
def schedule_model():
    return "success_schedule_model";

def get_current_path():
    return os.path.dirname(os.path.abspath(__file__))


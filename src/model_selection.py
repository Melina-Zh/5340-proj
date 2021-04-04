from config import *
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import datetime, pickle as pkl, numpy as np, matplotlib, pandas as pd, pymc3 as pm

from shared_utils import *
from sampling_utils import *
from matplotlib import rc
from collections import OrderedDict
import isoweek
import gc
import matplotlib.patheffects as PathEffects
from matplotlib.gridspec import SubplotSpec, GridSpec, GridSpecFromSubplotSpec
from BaseModel import BaseModel
import os

age_eastwest_by_name = dict(zip(["A","B","C"],combinations_age_eastwest))

with open('../data/counties/counties.pkl',"rb") as f:
    county_info = pkl.load(f)

best_model = {}
for disease in diseases:
    print("Evaluating model for {}...".format(disease))
    if disease=="borreliosis":
       prediction_region = "bavaria"
    else:
       prediction_region = "germany"
       
    data = load_data(disease, prediction_region, county_info)
    data_train, target_train, data_test, target_test = split_data(data)
    tspan = (target_train.index[0],target_train.index[-1])
    waics = {}
    for (name,(use_age,use_eastwest)) in age_eastwest_by_name.items():
        if disease=="borreliosis":
           use_eastwest = False
        # load sample trace
        trace = load_trace(disease, use_age, use_eastwest)
        
        # load model
        model = load_model(disease, use_age, use_eastwest)
        
        with model:
            waics[name] = pm.waic(trace).WAIC
            
    # do model selection
    best_key = min(waics, key=waics.get)
        
    use_age, use_eastwest = age_eastwest_by_name[best_key]
    if disease=="borreliosis":
        use_eastwest = False
    best_model[disease] = {"name": best_key, "use_age": use_age, "use_eastwest": use_eastwest, "comparison": waics}

with open('../data/comparison.pkl',"wb") as f:
    pkl.dump(best_model, f)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import visualize_data as vd
import math
import csv
import os

import lasso

l1_penalty = [0.05, 0.1, 0.3, 0.6, 0.8] # coefficient of penalty of weights


directory = 'Models/lasso_firstsim/'

for i in range(5):
    lasso.main(l1_penalty[i], i)
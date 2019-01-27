import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

import missingno as msno

import gc
import datetime
color = sns.color_palette()

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)

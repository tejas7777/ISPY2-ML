import numpy as np
import pandas as pd
from regression.models.linear import LinearRegression


df = pd.read_excel('dataset/train.xls')

df.rename(columns={'RelapseFreeSurvival (outcome)':'rfs'},inplace=True)
df = df.drop(columns=['ID'])

Linear_Regression = LinearRegression(data = df, target='rfs')
Linear_Regression.feature_select(method='lasso')

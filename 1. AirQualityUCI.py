'''
http://archive.ics.uci.edu/ml/datasets/Air+Quality

Attribute Information:

0 Date	(DD/MM/YYYY) 
1 Time	(HH.MM.SS) 
2 True hourly averaged concentration CO in mg/m^3 (reference analyzer) 
3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)	
4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer) 
5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer) 
6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)	
7 True hourly averaged NOx concentration in ppb (reference analyzer) 
8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)	
10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)	
11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted) 
12 Temperature in Â°C	
13 Relative Humidity (%) 
14 AH Absolute Humidity

Missing values are tagged with -200 value.
'''
# SLR model for Temprature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Loading dataset
dataset=pd.read_excel('ML AirQualityUCI.xlsx')

# Selecting X and y
X=dataset.iloc[:,2:12].values# selecting columns for X
y=dataset.iloc[:,[12]].values # selecting columns for y Temprature
y1=dataset.iloc[:,[13]].values # selecting columns for y Relative humidity
y2=dataset.iloc[:,[14]].values # selecting columns for y Absolute humidity

# Replacing Misssing values
from sklearn.impute import SimpleImputer
sim = SimpleImputer(missing_values=-200, strategy='mean')
X = sim.fit_transform(X)

# Removing rows where y is missing as their result would be misleading
missin_value_rows=np.where(y==-200) # Geeting rows with missing values
missin_value_rows[0]
y=np.delete(y,missin_value_rows[0]) # deleting from y
y1=np.delete(y1,missin_value_rows[0]) # deleting from y1
y2=np.delete(y2,missin_value_rows[0]) # deleting from y2
X=np.delete(X,missin_value_rows[0], axis=0) # Deleting from X also axis is given as it's a matrix


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression() # LR for y
lin_reg1=LinearRegression() # LR for y1
lin_reg2=LinearRegression() # LR for y2

lin_reg.fit(X,y)
lin_reg1.fit(X,y1)
lin_reg2.fit(X,y2)


lin_reg.score(X,y) #Score for Temp
#  0.6169310898748224
lin_reg1.score(X,y1) #Score for RH
#  0.4452752611896203
lin_reg2.score(X,y2) #Score for AH
# 0.7845955962007295

lin_reg.intercept_
#array([9.77659951]) or C
lin_reg.coef_

len(lin_reg.coef_)









































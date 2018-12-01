# C. Drew    ///   Final Exam, November 2018
# EE-629 Internet of Things   ///   Professor Kevin Lu
# Stevens Institute of Technology, Hoboken NJ
# TLDR: Take a CSV of Time-CPU%-Temp data and analyze it

import matplotlib.pyplot as plt
import numpy as np
import pylab as P
import pdb #what is this again, is it needed?
from pandas import * # "*" means import anything called below in the code
from datetime import datetime

# Import the Data from the CSV
data = read_csv('data.csv')
print(type(data))
print (data)
cpu  = data['CPU Usage %']
temp = data['Temperature C']
print(type(cpu))
# time = np.array([datetime.datetime(data['Time'])])

# Histogram of CPU %
numBins = 100
plt.hist(cpu, numBins, density=1, facecolor='blue', alpha=0.6 )
plt.title('Histogram of CPU Use % Probability')
plt.xlabel('CPU Use [%]')
plt.ylabel('Probability')
plt.subplots_adjust(left=0.2) # tweak spacing for y-label
plt.show()

# Histogram of Temp
numBins = 30
plt.hist(temp, numBins, density=1, facecolor='red', alpha=0.6 )
plt.title('Histogram of CPU Temperature Probability')
plt.xlabel('Temperature [˚C]')
plt.ylabel('Probability')
plt.subplots_adjust(left=0.2) # tweak spacing for y-label
plt.show()

# NOTE: TODO: Include title and labels and legend

# Time Series of Time-vs-CPU
t = np.arange(len(cpu))
plt.figure()
plt.plot(t,cpu)
plt.title('CPU Use -vs- Time')
plt.xlabel('Time [Absolute]')
plt.ylabel('CPU Use [%]')
plt.show()

# Time Series of Time-vs-Temperature
# N O T E: use regex to sanitize input CSV, as gaps or NaNs will crash boxplot
# or, in loop PC "for i in cpu, if i is a NaN then delete row"
plt.figure()
plt.boxplot(cpu,True)
plt.ylabel('CPU Usage [%]')
plt.title('Box Plot of CPU Usage')
plt.grid()
plt.show()

# Box Plot of Temperature
plt.figure()
plt.boxplot(temp,True)
plt.ylabel('Temperature [˚C]')
plt.title('Box Plot of Temperature')
plt.grid()
plt.show()

# Scatter Plot with Linear Regression

# Cross-validation prediction with temperature as target

# l = [slope * i + intercept for i in x]

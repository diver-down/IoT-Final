# C. Drew    ///   Final Exam, November 2018
# EE-629 Internet of Things   ///   Professor Kevin Lu
# Stevens Institute of Technology, Hoboken NJ
# TLDR: Take a CSV of Time-CPU%-Temp data and analyze it

import matplotlib.pyplot as plt
import numpy as np
import pylab as P
import pdb #what is this again, is it needed?
from pandas import * # "*" means import anything called below in the code
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from datetime import datetime

# Import the Data from the CSV
data = read_csv('data.csv')
cpu  = data['CPU Usage %']
temp = data['Temperature C']

# Histogram of CPU %
plt.figure()
numBins = 100
plt.hist(cpu, numBins, density=1, facecolor='blue', alpha=0.6 )
plt.title('Histogram of CPU Use % Probability')
plt.xlabel('CPU Use [%]')
plt.ylabel('Probability')
plt.subplots_adjust(left=0.2) # tweak spacing for y-label

# Histogram of Temp
plt.figure()
numBins = 30
plt.hist(temp, numBins, density=1, facecolor='red', alpha=0.6 )
plt.title('Histogram of CPU Temperature Probability')
plt.xlabel('Temperature [˚C]')
plt.ylabel('Probability')
plt.subplots_adjust(left=0.2) # tweak spacing for y-label

# Time Series of Time-Temp-vs-CPU
t = np.arange(len(cpu))
plt.figure()
plt.plot(t,cpu)
plt.plot(t,temp)
plt.legend(["CPU Use [%]","CPU Temperature [˚C]"])
plt.title('CPU Use % and Temperature -vs- Time')
plt.xlabel('Time [Seconds x 5]')
plt.ylabel('Value')

# N O T E: use regex to sanitize input CSV, as gaps or NaNs will crash boxplot
# or, in loop PC "for i in cpu, if i is a NaN then delete row"

# Box Plot of CPU
plt.figure()
plt.boxplot(cpu, True, 'rs', False)
plt.xlabel('CPU Usage [%]')
plt.title('Box Plot of CPU Usage')

# Box Plot of Temperature
plt.figure()
plt.boxplot(temp, True, 'rs', False)
plt.xlabel('Temperature [˚C]')
plt.title('Box Plot of Temperature')

# Scatter Plot with Linear Regression
plt.figure()
m,b = P.polyfit(cpu, temp, 1) # first-order polynomial fit
plt.plot(cpu, temp, 'bo', cpu, m*cpu+b, '--r')
plt.title('Scatter Plot of CPU Use -vs- Temperature')
plt.xlabel('CPU Use [%]')
plt.ylabel('Temperature [˚C]')
plt.grid()

# Cross-validation prediction with temperature as target
#plt.figure()
lr = linear_model.LinearRegression()
print('Number of instances: %d' % (cpu.shape[0]))
dataStack = (np.column_stack((cpu[:150],temp[:150])))
print("Data Stack:")
print(dataStack)
predicted = cross_val_predict(lr, dataStack, cpu[:150], cv=3)
fig, ax = plt.subplots()  # unsure if this CV is done properly
ax.scatter(temp[:150], predicted)
ax.plot([0, 100], [0, 100], 'k-', lw=2)
# alternate:
# ax.plot([temp.min(), temp.max()], [temp.min(), temp.max()], 'k-', lw=2)
ax.set_ylim(bottom = 0, top = 100)
plt.title('Cross-Validation Predition for Temperature')
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.show()

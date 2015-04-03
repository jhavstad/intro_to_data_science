import numpy as np
import pandas
import scipy
import scipy.stats

import time

from sklearn import decomposition
from datetime import datetime
#from ggplot import *
import matplotlib.pyplot as plt

'''
This is my Udacity Intro to Data Science Final project implementation of the analysis of the NYC MTA data
This should include:
1. A machine algorithm for data exploration (i.e. linear regression with gradient descent) - My approach
   is to use PCA (SVD) to extract relevant features from the data to create predictions about the number
   of entries per hour for the MTA turnstiles
2. A MapReduce implementation - My approach is to use MapReduce to reformat the dataframes that do not
   contain numerical data, e.g. data and time fields
3. A statistical analysis of the data for data confirmation (i.e. Mann Whitney, etc.) - My approach is to
   use the Mann Whitney rank test used in the class but applied to different data fields, including plots
'''

def create_hist_plot_data(plt, dataframe, constraint_field_label, constraint_function, value_field_label, plot_color, plot_label):
   '''
   You are passed in a dataframe called dataframe. 
   Use dataframe along with ggplot to make a data visualization
   focused on the MTA and weather data we used in assignment #3.  
   You should feel free to implement something that we discussed in class 
   (e.g., scatterplots, line plots, or histograms) or attempt to implement
   something more advanced if you'd like.  

   Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station
     * Which stations have more exits or entries at different times of day

   If you'd like to learn more about ggplot and its capabilities, take
   a look at the documentation at:
   https://pypi.python.org/pypi/ggplot/
     
   You can check out:
   https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
     
   To see all the columns and data points included in the dataframe 
   dataframe. 
     
   However, due to the limitation of our Amazon EC2 server, we are giving you about 1/3
   of the actual data in the dataframe dataframe
   '''
   # your code here
   df = dataframe[constraint_function(dataframe[constraint_field_label])][value_field_label]
   #df.plot(kind='hist', color=plot_color, bins=50, alpha=0.50, legend=True)
   df.hist(color=plot_color, bins=50, alpha=0.50, range=(0,6000), label=plot_label)
   #df.plot(legend=True)
   return plt

def create_box_plot_data(plt, dataframe, constraint_field_label, constraint_function, value_field_label, plot_color, plot_label):
   df = pandas.DataFrame(dataframe[constraint_function(dataframe[constraint_field_label])][value_field_label])
   df[value_field_label] = np.log10(df[value_field_label])
   print(df[value_field_label])
   df.boxplot()

def normalize_features(array):
   """
   Normalize the features in our data set.
   """
   mu = array.mean()
   sigma = array.std()
   # TODO: Check for the obligatory division by 0, i.e. is sigma 0
      
   array_normalized = (array-mu)/sigma

   return array_normalized, mu, sigma

def compute_cost(features, values, theta):
   """
   Compute the cost function given a set of features / values, and the values for our thetas.
    
   This should be the same code as the compute_cost function in the lesson #3 exercises. But
   feel free to implement your own.
   """
    
   # your code here
   # Cost function from previous lesson
   m = len(features)
   cost = 1. / (2. * m) * ((np.dot(features, theta) - values) ** 2).sum()

   return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
   """
   Perform gradient descent given a data set with an arbitrary number of features.
    
   This is the same gradient descent code as in the lesson #3 exercises. But feel free
   to implement your own.
   """
   m = len(values) * 1.0
   cost_history = []
   for i in range(num_iterations):
      # your code here
      cost_history.append(compute_cost(features, values, theta))
      theta += alpha * 1/m * np.dot((values - np.dot(features, theta)), features)
   return theta, pandas.Series(cost_history)
   
def extract_features(dataframe, y_field, columns_to_exclude=None):
   '''
   Extract the features used in making predictions from the original data set.
   '''
   values = pandas.DataFrame(dataframe[y_field], index=dataframe.index)
   if columns_to_exclude == None or len(columns_to_exclude) == 0:
      features = dataframe.drop([y_field], axis=1)
   else:
      features = dataframe.drop([y_field] + columns_to_exclude, axis=1)
    
   return values, features

def imputation(dataframe):
   '''
   This performs imutation on missing values or otherwise bad data.
   '''
   dataframe_new = dataframe.copy()
   imputed_values = []
   # Iterate over all the values in a specific dataframe, and
   # the dataframe is expected to be a pandas.Series object
   for index, values in dataframe_new.iteritems():
      for value in values:
         # This boolean expression returns True if and only if
         # the value being compared is non-null or non-NaN
         if value == value:
            imputed_values.append(value)
   # This is an error handling expression
   # If all values found are null or NaN then our array is
   # empty, as should be expected.  Simply set these values
   # to 1.  NOTE:  I seriously doubt if this is correct, and
   # there is probably a much better way of handling this type
   # of error.
   if len(imputed_values) == 0:
      imputed_value_avg = 1.0
   else:
      imputed_value_avg = np.mean(np.array(imputed_values))
   dataframe_new = dataframe_new.fillna(imputed_value_avg)
   
   for col in dataframe_new.columns:
      dataframe_new[col] = replace_zeros(dataframe_new[col])
   
   return dataframe_new

def replace_zeros(feature):
   '''
   This replaces any real valued 0's with a substantially small value.
   '''
   # Replace any zeros in the feature array with a substantially
   # small value
   feature_non_zeros = feature.replace(0.0, 1e-6)
   return feature_non_zeros

def enumerate_feature(feature):
   '''
   Attempt to enumerate the feature to a numerical format.
   '''
   feature_enum = feature.copy()
   
   # Try to parse object as a date object
   try:
      for index, value in feature.iteritems():
         date = datetime.strptime(value, '%Y-%m-%d')
         epoch = datetime(date.year, 1, 1)
         feature_enum[index] = date.toordinal() - epoch.toordinal() + 1
   except Exception as e:
      #print 'Not a date object: ' + str(e)
      pass
   else:
      #print 'Parsed date object'
      return feature_enum
   
   # Try to parse object as a time object
   try:
      for index, value in feature.iteritems():
         time = datetime.strptime(value, '%H:%M:%S')
         feature_enum[index] = time.hour * 3600 + time.minute * 60 + time.second
   except Exception as e:
      #print 'Not a time object: ' + str(e)
      pass
   else:
      #print 'Parsed a time object'
      return feature_enum
   
   # Try to parse object as a category
   try:
      categories = list()
      for index, value in feature.iteritems():
         if value not in categories:
            categories.append(value)
      for category in categories:
         feature_enum = feature_enum.replace(category, categories.index(category) + 1)
   except Exception as e:
      #print 'Not a category object: ' + str(e)
      pass
   else:
      #print 'Parsed a category object'
      return feature_enum
   
   return feature_enum
   
def enumerate_features(features):
   '''
   Determine which individual dataframes are not inherently numerical.
   '''
   # Iterate over each column in the whold dateframe
   features_new = features.copy()
   for col in features_new.columns:
      # If the feature is an object data type than attempt to enumerate its values
      if  features_new[col].dtype == np.dtype(np.object_):
         #print col + ' is an object data type'
         features_new[col] = enumerate_feature(features_new[col])
         
   return features_new

def predictions_linear_regression(values, features):
   '''
   This performs linear regression on the input features to generate
   predictions for the input values.
   '''
   #
   # Your implementation goes here. Feel free to write additional
   # helper functions
   #
   # Attempt a least squares approach to predicting the ridership values

   # To correctly perform the linear algebra equations we must transpose
   # the features array to access each attribute as row vs. a column
   # NOTE: We need to transpose the features array because the coefficient
   # matrix must align with x, not b - e.g. Ax = b
   features_array = np.array(features.transpose())
   values_array = np.array(values).flatten()
   
   # NOTE: Below is my implementation of Least Square
   # Fortunately, numpy has an implementation of least squares as well!
   A = np.zeros((len(features_array)+1, len(features_array)+1)) # The array of coefficients
   b = np.zeros(len(features_array)+1) # The array 
   n = len(features_array)
   k = len(features_array)
    
   A[0][0] = n
   b[0] = values_array.sum()
   for i in range(1, k+1):
      A[0][i] = features_array[i-1].sum()
      A[i][0] = features_array[i-1].sum()
      try:
         b[i] = np.dot(features_array[i-1], values_array).sum()
      except ValueError as ve:
         raise ve
      for j in range(1, k+1):
         try:
            A[i][j] = np.dot(features_array[i-1], features_array[j-1]).sum()
         except ValueError as ve:
            raise ve
            
   x = np.linalg.solve(A, b)
   print('Least squares x: ' + repr(x))
   predictions = x[0] * np.ones((1, len(values_array)))
   for i in range(1, k+1):
      predictions += np.dot(x[i], features_array[i-1])
    
   # The numpy implementation of least squares
   # NOTE: For some reason it is returning a prediction for a single frame, vs. an
   # entire series of values.  Investigate further.
   #predictions, residuals, rank, s = np.linalg.lstsq(features_array, values_array)
   predictions = pandas.DataFrame(predictions.transpose().flatten(), index=values.index, columns=values.columns)
        
   return predictions

def normalize_inputs(values, features):
   # Replace null or NaN values with the computed mean
   values = imputation(values)
   features = imputation(features)

   # Normalize the values and features
   values, vmu, vsigma = normalize_features(values)
   features, mu, sigma = normalize_features(features)

   return values, features
   
def predictions_gradient_descent(values, features):
   '''
   The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
   Using the information stored in the dataframe, lets predict the ridership of
   the NYC subway using linear regression with gradient descent.
    
   You can look at information contained in the turnstile weather dataframe 
   at the link below:
   https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
   Your prediction should have a R^2 value of .40 or better.
    
   Note: due to the memory and CPU limitation of our amazon EC2 instance, we will
   give you a random subet (~15%) of the data contained in turnstile_data_master_with_weather.csv
    
   If you receive a "server has encountered an error" message, that means you are hitting 
   the 30 second  limit that's placed on running your program. Try using a smaller number
   for num_iterations if that's the case.
    
   Or if you are using your own algorithm/modesl, see if you can optimize your code so it
   runs faster.
   '''
    
   m = len(values)

   #features['ones'] = np.ones(m)
   features_array = np.array(features)
   values_array = np.array(values).flatten()
    
   # Set values for alpha, number of iterations.
   alpha = 0.5 # please feel free to play with this value
   num_iterations = 75 # please feel free to play with this value

   #Initialize theta, perform gradient descent
   theta_gradient_descent = np.zeros(len(features.columns))
   theta_gradient_descent, cost_history = gradient_descent(features_array, values_array, theta_gradient_descent,
                                                            alpha, num_iterations)

   predictions = np.dot(features_array, theta_gradient_descent)
   
   predictions = pandas.DataFrame(predictions, index=values.index, columns=values.columns)

   return predictions

def feature_decomposition(features):
   '''
   This perfrom decomposition transform on the set of parameters passed as input.
   
   It returns a new set of features that is the decomposition of the input argument
   
   See the following link for more information:
   http://en.wikipedia.org/wiki/Principal_component_analysis
   '''
   print 'Features columns: ' + str(features.columns)
   pca = decomposition.PCA()
   pca.fit(features)
   features_transform = pandas.DataFrame(pca.transform(features), index=features.index, columns=features.columns)
   return features_transform
   
def mann_whitney_plus_means(dataframe, constraining_field_label, constraining_field_function1, constraining_field_function2, value_field_label):
   '''
   This function will consume the dataframe dataframe containing
   the input dataframe. 
    
   You will want to take the means and run the Mann Whitney U test on the 
   column with the value label in the dataframe dataframe.
    
   This function should return:
        1) the mean of entries with the constraint value > 0 
        2) the mean of entries with the constraint value = 0
        3) the Mann-Whitney U statistic and p-value comparing the value of the 
           field with the value label with the entries with and without a positive
           non-zero values of the field with the constraing field
    
   You should feel free to use scipy's Mann-Whitney implementation, and
   also might find it useful to use numpy's mean function.  
    
   Here are some documentations:
   http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
   http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    
   You can look at the final turnstile weather data at the link below:
   https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
   '''
   with_constraint_mean    = np.mean(dataframe[constraining_field_function1(dataframe[constraining_field_label])][value_field_label])
   without_constraint_mean = np.mean(dataframe[constraining_field_function2(dataframe[constraining_field_label])][value_field_label])
   U, p = scipy.stats.mannwhitneyu(x = dataframe[constraining_field_function1(dataframe[constraining_field_label])][value_field_label], y = dataframe[constraining_field_function2(dataframe[constraining_field_label])][value_field_label])
    
   return with_constraint_mean, without_constraint_mean, U, p
   
def compute_r_squared(data, predictions):
   '''
   In exercise 5, we calculated the R^2 value for you. But why don't you try and
   and calculate the R^2 value yourself.
    
   Given a list of original data points, and also a list of predicted data points,
   write a function that will compute and return the coefficient of determination (R^2)
   for this data.  numpy.mean() and numpy.sum() might both be useful here, but
   not necessary.

   Documentation about numpy.mean() and numpy.sum() below:
   http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
   http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
   '''
   
   data_array = np.array(data).flatten()
   predictions_array = np.array(predictions).flatten()
    
   # your code here
   r_squared = 1 - (((data_array - predictions_array) ** 2).sum() / ((data_array - data_array.mean()) ** 2).sum())
    
   return r_squared

def continue_program(prompt):
   key = raw_input((prompt + ' ([yY]es or [nN]no)? => '))
   
   valid_inputs = ['y', 'Y', 'n', 'N']
   
   while key == None or key not in valid_inputs:
      key = input('Please press [yY] for yes or [nN] for no. => ')
      
   return key

def user_quit(user_input):
   if user_input == 'n' or user_input == 'N':
      return True
   
   return False

def main():
   # Prompt user to begin program
   key = continue_program('Ready to begin')
   if user_quit(key):
      return
   
   # Step 1: Open the turnstile weather data file
   path = '../Data/'
   # The following is the filename for the original data set
   filename = 'turnstile_data_master_with_weather.csv'
   # The following is the filename for the improved data set
   #filename = 'turnstile_weather_v2.csv'
   turnstile_weather = None
   print 'Opening ' + path + filename + '.'
   try:
      turnstile_weather = pandas.read_csv(path + filename)
   except IOError as ioe:
      print str(ioe)
      print 'Fail!\n'
      return
   else:
      print 'Success!\n'
   
   # Step 2: Perform data confirmation using standard statistical test including creating plots 
   # Prompt user to begin data confirmation analysis
   key = continue_program('Ready to begin data confirmation analysis')
   if user_quit(key):
      return
   fig1_fname = 'turnstile_weather_rain_hist.png'
   print 'Creating data plot for rain: ' + fig1_fname + '.'
   plt.figure()
   create_hist_plot_data(plt, turnstile_weather, 'rain', lambda x: x == 0, 'ENTRIESn_hourly', 'green', 'No Rain')
   create_hist_plot_data(plt, turnstile_weather, 'rain', lambda x: x > 0, 'ENTRIESn_hourly', 'blue', 'Rain')
   plt.xlabel('Hourly Entries')
   plt.ylabel('Frequency')
   plt.title('Frequency of Hourly Entries for Rainy and Non-Rainy Days')
   plt.legend()
   plt.savefig(fig1_fname)
   print 'Success!'
   fig2_fname = 'turnstile_weather_rain_box.png'
   print 'Creating box data plot for days with no rain: ' + fig2_fname + '.'
   plt.figure()
   create_box_plot_data(plt, turnstile_weather, 'rain', lambda x: x == 0, 'ENTRIESn_hourly', 'yellow', 'No Rain')
   plt.xlabel('')
   plt.ylabel('Hourly Entries (log 10 scale)')
   plt.title('Box Plot of Hourly Entries for Non-Rainy Days')
   plt.legend()
   plt.savefig(fig2_fname)
   fig3_fname = 'turnstile_weather_no_rain_box.png'
   print 'Creating box data plot for days with rain: ' + fig3_fname + '.'
   plt.figure()
   create_box_plot_data(plt, turnstile_weather, 'rain', lambda x: x > 0, 'ENTRIESn_hourly', 'red', 'Rain')
   plt.xlabel('')
   plt.ylabel('Hourly Entries (log 10 scale)')
   plt.title('Box Plot of Hourly Entries for Rainy Days')
   plt.legend()
   plt.savefig(fig3_fname)
   print 'Success!\n'
   print 'Calculate Mann Whitney U test on rain...'
   mu1, mu2, U, p = mann_whitney_plus_means(turnstile_weather, 'rain', lambda x: x == 0, lambda x: x > 0, 'ENTRIESn_hourly')
   print 'Mean with rain: {0}, Mean without rain: {1}, U: {2}, p-value: {3}'.format(mu1, mu2, U, p)
   print 'Success!'
   print 'Calculating Mann Witney U test on fog...'
   mu1, mu2, U, p = mann_whitney_plus_means(turnstile_weather, 'fog', lambda x: x == 0, lambda x: x > 0, 'ENTRIESn_hourly')
   print 'Mean with fog: {0}, Mean without fog: {1}, U: {2}, p-value: {3}'.format(mu1, mu2, U, p)
   print 'Success!\n'
   
   print 'Features columns: ' + str(turnstile_weather.columns)
   
   # Step 3: Extract features from data set using iterative programming
   # Prompt user to begin extracting features from data set
   key = continue_program('Ready to extract and enumerate features for data analysis')
   if user_quit(key):
      return
   print 'Extracting features and values from input...'
   # Step 2: Separate the variables we want to predict from the others and massage the data
   #values, features = extract_features(turnstile_weather, 'ENTRIESn_hourly', ['Unnamed: 0'])
   values, features = extract_features(turnstile_weather, 'ENTRIESn_hourly')
   features = enumerate_features(features)

   # Replace null or NaN values with the computed mean
   values = imputation(values)
   features = imputation(features)

   # Normalize the values and features
   values, vmu, vsigma = normalize_features(values)
   features, mu, sigma = normalize_features(features)
   #if len(features.columns) != 20:
      #print 'Fail!\n'
      #return
      
   # Replace null or NaN values with the computed mean
   # I want to do this again because the normalization may have casued some round off errors
   # because values became too small, therefore becoming NaN
   values = imputation(values)
   features = imputation(features)
      
   print 'Success!\n'

   # Step 4: Perform data exploration using machine learning algorithms
   # Prompt user to begin data exploration analysis
   key = continue_program('Ready to begin data exploration analysis')
   if user_quit(key):
      return   
   # Optional: Perform feature decomposition
   print 'Performing PCA on features...'
   features = feature_decomposition(features)
   print 'Success!\n'
   
   print('# Values: ' + str(len(values)))
   print('# Features: ' + str(len(features)))
   
   print 'Performing gradient descent...'
   # 1st machine learning algorithm is gradient descent
   start = time.clock()
   predictions_gd = predictions_gradient_descent(values, features)
   end = time.clock()
   r_squared_gd = compute_r_squared(values, predictions_gd)
   print 'The R^2 value for gradient descent is: ' + str(r_squared_gd)
   print 'Time to complete model: ' + str(end - start) + ' seconds'
   print 'Success!\n'
   
   print 'Performing least squares...'
   # 2nd machine learning algorithm is linear regression
   predictions_ls = predictions_linear_regression(values, features)
   start = time.clock()
   r_squared_ls = compute_r_squared(values, predictions_ls)
   end = time.clock()
   print 'The R^2 value for least squares is: ' + str(r_squared_ls)
   print 'Time to complete model: ' + str(end - start) + ' seconds'
   print 'Success!\n'
   
   print 'Program has finished!'

if __name__ == '__main__':
   main()
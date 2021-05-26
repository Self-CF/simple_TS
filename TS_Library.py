#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:16:38 2021

@author: carlo-f33

Python file containig utility functions for time series modules, 
tested in separate notebooks - located in this same folder.
Used/called in Time series pipelines (a few in this folder) .

Note can either enlist alphabetically or by subject (or both: buth then keep
 one library per subject?)

**** BETTER - define classes like: filter, smoother, norm etc.

Each function is documented and a commented call instruction is provided.
(so can run as example in a notebook)
""" 


"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Moving Average (linear)
  receive: a panda series or df and the number of terms 
  returns: NA
  plots original and MA-ed signal
  
  see more general call at: https://www.geeksforgeeks.org/python-pandas-series-rolling/

"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_rolling(timeseries, kwin=3):
    rol_mean = timeseries.rolling(kwin).mean()
    rol_std = timeseries.rolling(kwin).std()
    
#
# interactive_plot(my_df, 'my_title')# normalize(tdf,20) 
#____________________________________________
#
# interactive_plot(my_df, 'my_title')
    fig = plt.figure(figsize = (12, 8))
    orig = plt.plot(timeseries, color = "blue", label = "Original")
    Rmean = plt.plot(rol_mean, color = "red", label = "Rolling Mean")
    Rstd = plt.plot(rol_std, color = "black", label = "Rolling Std")
    plt.legend(loc = "best")
    plt.title("Rolling Mean and Standard Deviation (window = "+str(kwin)+")")
    plt.show()

# Usage: uncomment to test, recall set kwin=depth, defaults to 3
# provide tseries dataset:
# from pandas import Series
# import pandas as pd
# tseries = pd.read_csv('~/Documents/data/sunspots.csv')
# plot_rolling(tseries,20) 
#____________________________________________



"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Exponentially Weighted Moving Average - is 50
# Note, ewm method applicable on pandas Series and DataFrame objects only
# alpha (not romeo):  0<α≤1, smoothing factor

"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_ewma(timeseries, alpha):
    expw_ma = timeseries.ewm(alpha=alpha).mean()

    fig = plt.figure(figsize = (12, 8))
    og_line = plt.plot(timeseries, color = "blue", label = "Original")
    exwm_line = plt.plot(expw_ma, color = "red", label = "EWMA")
    plt.legend(loc = "best")
    plt.title("EWMA (alpha= "+str(alpha)+")")
    plt.show()

# Usage: uncomment to test
# provide tseries dataset:
# from pandas import Series
# import pandas as pd
# tseries = pd.read_csv('~/Documents/data/sunspots.csv')
# plot_emwa(tseries,0.5)  # play with alpha
#____________________________________________





"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Savitzky - Golay; 
A filter that tries to fit adjacent datapoint windows with low-order polynomials
 (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)

# simple version, no derivatives etc. For func. optional params in scipy,
# cf https://scipy.github.io/devdocs/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter

window = the number of coefficients (odd)
# may want to parametrize the poly order based on signal info...

# in class, may want to explain benefits and drawbacks compared to -say - MA or other filters
# highlight the convolutional/correlation aspects

#
# interactive_plot(my_df, 'my_title')# normalize(tdf,20) 
#____________________________________________
#
# interactive_plot(my_df, 'my_title')
# better to start with a simple array. 
Need to care about conversions: the dreadful pandas has type conversion problem, form datetimes to Date differences 
# just horrid to do.

"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_SG(timeseries, window, npoly=3): # might pass optionally the order of the poly
    from scipy.signal import savgol_filter # https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    np.set_printoptions(precision=2)  # For compact display.
    
    savgol_filter(timeseries, window, 2)  # window= win size , 2= order of fitted polynomial, against default 3
    SG_mean = timeseries.savgol_filter(timeseries, window, npoly).mean()
    SG_std = timeseries.savgol_filter(timeseries, window, npoly).std()
    
    fig = plt.figure(figsize = (12, 8))
    og = plt.plot(timeseries, color = "blue", label = "Original")
    mean = plt.plot(SG_mean, color = "red", label = "S-G Mean")
    std = plt.plot(SG_std, color = "black", label = "S-G Std")
    plt.legend(loc = "best")
    plt.title("Savitzky-Golay Mean and Standard Deviation (window = "+str(window)+")")
    plt.show()
    
    

# Usage: uncomment to test
# provide tseries dataset, 
# from pandas import Series
# import pandas as pd
# tseries = pd.read_csv('~/Documents/data/sunspots.csv') 
# plot_SG(series, 5, 2) # recall: win length and poly order
#____________________________________________

 





"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Normalize passed variable/s based on initial value/s
  passed argument assumed to be dataset - hence has 'columns'
  NB first column is timebase hence cols start at 1

"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x

# Usage: 
# uncomment to test, provide dataset (dataframe)
# normalize(tdf,20) 
#____________________________________________






""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plot interactively with Plotly Express
input: dataset - with time base as first col; title (string) 
# worth re-engineering  all other ploit funcs, so to have al plots interactive

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# import plotly.express as px 

def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()

# Usage: 
# uncomment to test, provide dataset (dataframe)
# interactive_plot(my_df, 'my_title')






"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test Stationarity: Augmented Dickey-Fuller - ADF
# worth completing with other dynamical systems tools - in ad hoc Dyn Syst. lib
# recall that ADF tests the null hypothesis H0 that the autoregressive model
# has a unit root: if you seek stationarity, you want to reject H0.

# autolag=number of lags to minimize the info criterion (AIC, BIC, t-stat)
# cf https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
# for edu/ref, cf: http://www.ams.sunysb.edu/~zhu/ams586/UnitRoot_ADF.pdf
(use to identify the underlying time series model: do we have: ARMA, or 
 rend + ARMA, or ARIMA)...
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Test Stationarity
def ADF_test(timeseries):
    """"Augmented Dickey-Fuller Test
    Test for Stationarity"""
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    print("Results of Dickey-Fuller Test:")
    df_test = adfuller(timeseries, autolag = "AIC")  
    df_output = pd.Series(df_test[0:4]
                          index = ["Test Statistic", "p-value", "#Lags Used",
                                   "Number of Observations Used"])
    print(df_output)


# Usage: 
# uncomment to test, provide dataset (dataframe)
# Test application on the sunspots dataset
# ADF_test(tseries['MonthAvgSunspot'])

# recall interpretation, ex of above command:
"""    
Results of Dickey-Fuller Test:
Test Statistic                -1.049705e+01
p-value                        1.108552e-18
Lags Used                      2.800000e+01
Number of Observations Used    3.236000e+03
dtype: float64

We reject the null H0, as the probability of getting such a low p-value by 
random chance is extremely unlikely. Hence the series is stationary

"""






"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generalized function to perform transformation from TS to supervised ML
# by generating k new columns, m new targets (spaced by 1 - hence performing
#  an m horizon forecasting). k might be derived from PACF but alos looking 
# at the dynamical System project
# cf specific notebook with all the cases and the pots to test dependencies 
# on these params.

# the older first version was derived from Brainlee website: <link>
    
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ts_2_sml(data, n_in=1, n_out=1, dropnan=False):  # in original was True
    
    '''
    PURPOSE
    Transform from time series to supervised ML problem using time_lag - or Sliding Window method.
    Constructs and names the new columns by variable number + time step. 
    So we can design different time step sequence type forecasting problems from a given univariate or 
    multivariate time series.

    INPUT:
    data  = Sequence of observations as a list or 2D NumPy array. Required.
    n_in  = Number of lag observations as input (X). Values between [1..len(data)] Optional. Defaults to 1
    n_out = Number of observations as output (y). Values between [0..len(data)-1]. Optional. Defaults to 1
    drop_nan = drops nan values (not a number), default is true

    OUTPUT:
    Pandas DataFrame of series, framed for supervised learning task
    Once done, user may split the returned DataFrame into X and y components for supervised learning
    Default parameters will construct a DataFrame with t-1 as X; and t as y 
    
    NOTE: we may want to establish k, see section on ACF / PACF, close to the end of the doc
    '''
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:one
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # wrap up
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values; could be more sophisticated (e.g. imputing...)
    # but perhaps it's best to deal with that separately
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    
# Usage: simplest - several other examples in the ad-hoc project -
# see also extensions in  Dyn System project
# univariate, single step prediction, with three steps back
# here only using simple mock data generated for purpose
# # defaults params: n_in=1, n_out=1, dropnan=T
#values = [x for x in range(10)]   # here generate values for a 1-dim mock ts

# data = ts_2_sml(values)
#data = ts_2_sml(values, 4) # observations, number of lags we consider --> hence number of columns generated
#print(data)

#____________________________________________




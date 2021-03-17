import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pywt 

'''
#Importing stock data 
import yfinance as yf
from datetime import date,datetime,timedelta
ticker = '^GSPC'
first_day = datetime(2000, 1, 3)
last_day = datetime(2019, 7, 1)
data = yf.Ticker(ticker).history(interval = '1d', start=first_day, end=last_day)
data.reset_index(inplace=True)
'''

'''
#Importing our crypto data
ticker = 'QTUMUSD' #Try QTUMUSD, XBTEUR, ETCUSD, ZECXBT, GNOXBT, XBTEUR, LTCEUR, XBTUSD, EOSXBT, EOSETH, GNOUSD
data = pd.read_csv('/Users/Sanjit/Google Drive/CollectiWise/Data/high_low.csv') #change this
data = data[data['asset'] == ticker]
data.reset_index(inplace=True, drop=True)
'''

data = pd.read_csv('/Users/Sanjit/Repos/CollectiWise/formatted_features.csv')
column = 'e_XBTUSD avg_price'
data = data[column]

from waveletDenoising import denoise, SNR, RMSE, optDenoise, standardise, gridSearch_v2, optDenoise_v2 #Store this file in the same folder as 'waveletDenoising.py'
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.restoration import denoise_wavelet

#x = np.array(data.Close)
x = np.array(data)
original_mean = np.mean(x)
original_std = np.std(x)

#In the paper they used zero-mean normalization, which means the series is just shifted vertically downwards by its mean.
x = x - np.mean(x) #equivalently, standardise(x, 0, np.std(x))

#x = standardise(x, 0, 1) #N(0,1) standardisation

#See https://www.youtube.com/watch?v=HSG-gVALa84 
#y = denoise_wavelet(x, wavelet='coif3', mode='hard', wavelet_levels=3, method='BayesShrink', rescale_sigma=True)
#method: 'BayesShrink' or 'VisuShrink'
#Most of the time, the denoised series is basically identical to the original. Problem is worse when we standardise to N(0, 1)
#VisuShrink doesn't capture price peaks, and these obviously can't be noise.

y = optDenoise_v2(x) 

#x = x + original_mean
#y = y + original_mean
#x = standardise(x, original_mean, original_std)
#y = standardise(x, original_mean, original_std)

print("SNR: ", SNR(x, y))
print("RMSE: ", RMSE(x, y))

plt.plot(data.index, x, color='Green')
plt.plot(data.index, y, color='Red')
#plt.title(ticker)
plt.title(column)
plt.show()


'''
We see strange behaviour when the prices are very large (XBTEUR, XBTUSD, in 1000s) and very small (GNOXBT, EOSXBT, in 0.001s)
When prices are large, the denoised signal is almost identical to the raw signal
When prices are small, the denoised signal is a constant zero signal, i.e. nothing like the raw signal

It seems that in the second case, everything is considered noise since all the movements are so small, and in the first case,
nothing is considered noise since all the movements are so large. 

There must be some way to 'normalise' the data, so that the absolute value of prices moves is irrelevant, and only the relative
value of price moves matters. 

I've now implented this in the rescale function: it rescales the data to have any mean and std you specify. The issue with 
rescaling and then descaling is that RMSE increases by a lot (for GSPC, where new_mean = sqrt(old_mean) and similarly for std). 
Despite this, the plot looks alright. 

Why do we descale? At some point we need to, either after feeding the data through the model or before.
Rescaling, to the squares of the orignial mean and standard deviation, works really nicely with QTUMUSD. 
When the numbers are too small (<1), there seems to be some kind of numerical overflow: the denoised signal is way off. So, the
usual mean = 0 std = 1 transform is not really an option. 
Many cryptos were worth extremely small amounts when they started trading. In these cases, the denoised signal at the start of the
period is way off. ZECXBT offers basically no information. 

It seems that it's not easy to write one function which can properly denoise every series we give it in just one click. 
There needs to be an element of inspection. Maybe we can try a grid search for each series, but I don't see anything better.

I have now implemented a grid search! Don't see how we can do much better. It works for the most part, but for certain assets,
the denoised series is still not right. 
'''
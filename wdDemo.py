import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pywt 

#Importing stock data 
import yfinance as yf
from datetime import date,datetime,timedelta
ticker = '^GSPC'
first_day = datetime(2000, 1, 3)
last_day = datetime(2019, 7, 1)
data = yf.Ticker(ticker).history(interval = '1d', start=first_day, end=last_day)
data.reset_index(inplace=True)

'''
#Importing our crypto data
ticker = 'QTUMUSD' #Try QTUMUSD, XBTEUR, ETCUSD, ZECXBT, GNOXBT, XBTEUR, LTCEUR, XBTUSD, EOSXBT, EOSETH, GNOUSD
data = pd.read_csv('/Users/Sanjit/Google Drive/CollectiWise/Data/high_low.csv') #change this
data = data[data['asset'] == ticker]
data.reset_index(inplace=True, drop=True)
'''

from waveletDenoising import denoise, SNR, RMSE #Store this file in the same folder as 'waveletDenoising.py'
x = data.Close
y = denoise(x)
print("SNR: ", SNR(x, y))
print("RMSE: ", RMSE(x, y))

plt.plot(data.index, x, color='Green')
plt.plot(data.index, y, color='Red')
plt.title(ticker)
plt.show()

'''
We see strange behaviour when the prices are very large (XBTEUR, XBTUSD, in 1000s) and very small (GNOXBT, EOSXBT, in 0.001s)
When prices are large, the denoised signal is almost identical to the raw signal
When prices are small, the denoised signal is a constant zero signal, i.e. nothing like the raw signal

It seems that in the second case, everything is considered noise since all the movements are so small, and in the first case,
nothing is considered noise since all the movements are so large. 

There must be some way to 'normalise' the data, so that the absolute value of prices moves is irrelevant, and only the relative
value of price moves matters. 
'''
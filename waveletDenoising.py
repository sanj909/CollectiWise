import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pywt #after running pip install PyWavelets (https://github.com/PyWavelets/pywt, https://pywavelets.readthedocs.io/en/latest/)
from sklearn.preprocessing import StandardScaler
from skimage.restoration import denoise_wavelet

'''
The denoising steps are the following : https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
(1)Apply the dwt to the signal
    Which signal extension mode is best for financial time series? 
    See https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes
    Periodization is bad: visually inspect the start of the signal constructed with this mode
    Actually it makes barely any difference, after trying out a few. 
    if level is not specified or level = None, it's calculated using the dwt_max_level function. For this data, it is 5.
    coeffs is an array of arrays. len(coeffs) = level + 1
    As level increases the denoised signal gets smoother and is more different from the original signal.

(2)Compute the threshold corresponding to the chosen level
    Increasing threshold seems to smooth out the curve more 
    I've no idea how thresholds are calculated. See https://uk.mathworks.com/help/wavelet/ref/thselect.html
    threshold = 1 gives great results in terms of a high SNR and low RMSE.
    But not denoising at all gives the best result with these metrics (infinite SNR, 0 RMSE)! These metrics seem useless.
        I've been assessing performance visually, balancing smoothness of the curve with the similarity to the original.
    So, these (wavelet, mode, level) are true hyperparameters, which require retraining and testing of ML model to optimise
    https://uk.mathworks.com/help/wavelet/ug/denoising-signals-and-images.html: 
        'Minimax and SURE threshold selection rules are more conservative and would be more convenient when small details of 
        the signal lie near the noise range. The two other (sqtwolog, mixture of sqtwolog and SURE) rules remove the noise 
        more efficiently'
    https://uk.mathworks.com/help/wavelet/ug/wavelet-denoising.html:
        We assume that the magnitude of noise is constant throughout the signal. MATLAB has a function which automatically 
            relaxes this assumption, i.e. automatically adjusts the threshold depending on the level of noise at each segment
            of the series (no of segments are also calculated automatically) but I'm not sure how it works. If you have MATLAB,
            this is one way we can improve the model. 
        Maybe we can just split every series into 5 parts, denoise each separately, and stitch them back together. This will 
        only work with a different threshold function though: sqtwolog, which we're using now, depends only on the length of 
        the series. 

(3)Only keep coefficients with a value higher than the threshold
    Which threshold mode is best for financial time series? 
    hard thresholding is much better the soft, by visual inspection of plot. garrote is similar to hard.
    It seems that they used hard thresholding in the paper.

(4)Apply the inverse dwt to retrieve the signal
(5)Sometimes, signal length is 1 greater than raw data length. We'll just remove the last value for now.
'''

#Input must be a numpy array. Output is a numpy array.
def denoise(raw, wavelet, level, mode='symmetric'):
    coeffs = pywt.wavedec(raw, wavelet, mode=mode, level=level) #(1)
    threshold = np.sqrt(2*np.log(len(raw))) #(2)sqtwolog function in MATLAB
    coeffs = [pywt.threshold(i, value=threshold, mode='hard') for i in coeffs]#(3)
    signal = pywt.waverec(coeffs, wavelet, mode=mode)#(4)
    if len(signal) > len(raw):#(5)
        signal = np.delete(signal, -1)
    return signal

def SNR(raw, denoised): #returns signal-to-noise ratio; equation (9) in paper; xhat_j is presumably the denoised series
    num = np.sum(np.power(raw, 2))
    den = np.sum(np.power(raw - denoised, 2))
    return 10*np.log(num/den)

def RMSE(raw, denoised): #Google 'root mean square deviation' for formula; equation (12) in paper is incorrect
    ss = np.sum(np.power(raw - denoised, 2))
    return np.sqrt(ss)

#https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
def standardise(x, new_mean, new_std):
    return new_mean + (x - np.mean(x))*(new_std/np.std(x))

#Rescaling series to ensure consistent performance of denoising function
#The new mean should roughly be between 10 and 100, for most assets, according to the block below.
#denoise function doesn't work with mean 0 variance 1 data for some reason
def rescale(x, orgnl_mean, orgnl_std):
    if 1 < orgnl_mean <= 10:
        x = standardise(x, np.power(orgnl_mean, 2), np.power(orgnl_std, 2))
    elif 100 < orgnl_mean:
        x = standardise(x, np.sqrt(orgnl_mean), np.sqrt(orgnl_std))
    elif orgnl_mean < 1:
        x = standardise(x, np.power(100, orgnl_mean), np.power(100, orgnl_std))
    elif orgnl_mean < 0.1:
        x = standardise(x, np.power(10000, orgnl_mean), np.power(10000, orgnl_std))
    return x

def gridSearch(x, orgnl_mean, orgnl_std):
    result = [-100000, '', 0] #SNR - RMSE, wavelet, level
    for w in pywt.wavelist(kind='discrete'):
        for l in range(2, 5):
            #x = rescale(x, orgnl_mean, orgnl_std)
            x = standardise(x, 0, 1)
            y = denoise(x, w, l)

            #x = standardise(x, orgnl_mean, orgnl_std)
            #y = standardise(y, orgnl_mean, orgnl_std)

            if (SNR(x, y) - RMSE(x, y)) > result[0]:
                result[0] = (SNR(x, y) - RMSE(x, y)); result[1] = w; result[2] = l

    return result

#Input must be a simple iterable e.g. np.array, pd.Series, array. Output is a numpy array.
def optDenoise(x):
    x = np.array(x)
    orgnl_mean = np.mean(x); orgnl_std = np.std(x)

    params = gridSearch(x, orgnl_mean, orgnl_std)
    #x = rescale(x, orgnl_mean, orgnl_std)
    y = denoise(x, params[1], params[2])

    #standardise back to original distribution
    #x = standardise(x, orgnl_mean, orgnl_std)
    #y = standardise(y, orgnl_mean, orgnl_std)

    return y

#grid search best parameters for denoising function. 
def gridSearch_v2(x, metric):
    #metric=1: maximise SNR - RMSE
    #metric=2: maximise SNR
    #metric=3: minimise RMSE
    result = ['', 0, '', '', 1000000, 0, -1000000] #wavelet, level, mode, method, RMSE, SNR, SNR-RMSE

    #Only consider haar, db, sym, coif wavelet basis functions, as these are relatively suitable for financial data
    for w in [wavelet for wavelet in pywt.wavelist(kind='discrete') if wavelet.startswith(('haar', 'db', 'sym', 'coif'))]:
        for l in range(1, 5):
            for m in ['hard', 'soft']:
                for method in ['BayesShrink', 'VisuShrink']:
                    y = denoise_wavelet(x, wavelet=w, mode=m, wavelet_levels=l, method=method, rescale_sigma=True)

                    snr = SNR(x, y)
                    rmse = RMSE(x, y)

                    if metric == 1:
                        if (snr - rmse) > result[6]:
                            result[6] = (snr - rmse); result[0] = w; result[1] = l; result[2] = m; result[3] = method
                    elif metric == 2:
                        if (snr) > result[5]:
                            result[5] = (snr); result[0] = w; result[1] = l; result[2] = m; result[3] = method
                    elif metric == 3:
                        if (rmse) < result[4]:
                            result[4] = (rmse); result[0] = w; result[1] = l; result[2] = m; result[3] = method

    return result

def optDenoise_v2(x):
    x = np.array(x)
    #original_mean = np.mean(x)

    #In the paper they used zero-mean normalization, which means the series is just shifted vertically downwards by its mean.
    #x = x - np.mean(x) #equivalently, standardise(x, 0, np.std(x))

    #grid search best parameters for denoising function. 
    #maximise SNR-RMSE, as they recommended in the paper.
    params = gridSearch_v2(x, 1) 

    #See https://www.youtube.com/watch?v=HSG-gVALa84 
    y = denoise_wavelet(x, wavelet=params[0], wavelet_levels=params[1], mode=params[2], method=params[3], rescale_sigma=True)
    #y = denoise_wavelet(x, wavelet='coif3', wavelet_levels=3, mode='hard', method='BayesShrink', rescale_sigma=True) #paramters used in paper

    '''
    method: 'BayesShrink' or 'VisuShrink'
    Most of the time, the denoised series is basically identical to the original
    #  VisuShrink doesn't capture price peaks, and these obviously can't be noise.
    '''
    #y = y + original_mean
    
    return y

#takes a numerical dataframe as input
#treats each column as a series, denoises each of these series
#stitches back together and returns a dataframe
def denoise_df(df):
    for column in df.columns:
        x = np.array(df[column])
        df[column] = optDenoise_v2(x)
    return df
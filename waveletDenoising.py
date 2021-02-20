import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pywt #after running pip install PyWavelets (https://github.com/PyWavelets/pywt, https://pywavelets.readthedocs.io/en/latest/)

'''
The denoising steps are the following : https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
(1)Apply the dwt to the signal
    Which signal extension mode is best for financial time series? 
    See https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes
    Periodization is bad: visually inspect the start of the signal constructed with this mode
    Actually it makes barely any difference, after trying out a few. 
    if level is not specified or level = None, it's calculated using the dwt_max_level function. For this data, it is 5.
    coeffs is an array of arrays. len(coeffs) = level + 1

(2)Compute the threshold corresponding to the chosen level
    Increasing threshold seems to smooth out the curve more 
    I've no idea how thresholds are calculated. See https://uk.mathworks.com/help/wavelet/ref/thselect.html
    threshold = 1 gives great results in terms of a high SNR and low RMSE.
    But not denoising at all gives the best result with these metrics (infinite SNR, 0 RMSE)! These metrics seem useless.
    So, these (wavelet, mode, level) are true hyperparameters, which require retraining and testing of ML model to optimise
    https://uk.mathworks.com/help/wavelet/ug/denoising-signals-and-images.html: 
        'Minimax and SURE threshold selection rules are more conservative and would be more convenient when small details of 
        the signal lie near the noise range. The two other (sqtwolog, mixture of sqtwolog and SURE) rules remove the noise 
        more efficiently'

(3)Only keep coefficients with a value higher than the threshold
    Which threshold mode is best for financial time series? 
    hard thresholding is much better the soft, by visual inspection of plot. garrote is similar to hard.
    It seems that they used hard thresholding in the paper.

(4)Apply the inverse dwt to retrieve the signal
(5)Sometimes, signal length is 1 greater than raw data length. We'll just remove the last value for now.
'''

#Input must be an iterable like a list, pandas.Series, or numpy.array. Output is a numpy array.
def denoise(raw, mode='symmetric', wavelet='coif3'):
    coeffs = pywt.wavedec(raw, wavelet, mode=mode, level=4) #(1)
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
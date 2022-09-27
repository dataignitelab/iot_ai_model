#%%  Define class for time feature extraction##
import numpy as np
from scipy.stats import skew, kurtosis

class Extract_Time_Features:    
    def __init__(self, rawTimeData):
        self._TimeFeatures = []
        self._rawTimeData = rawTimeData

    def AbsMax(self):
        self._absmax = np.abs(self._rawTimeData).max(axis=1)
        return self._absmax
    
    def AbsMean(self):
        self._absmean = np.abs(self._rawTimeData).mean(axis=1)
        return self._absmean
    
    def P2P(self):
        self._peak2peak = np.max(self._rawTimeData,axis=1) - np.min(self._rawTimeData,axis=1)
        return self._peak2peak 
    
    def Skewness(self):
        self._skewness = skew(self._rawTimeData, axis=1, nan_policy='raise')
        return self._skewness
    
    def Kurtosis(self):
        self._kurtosis = kurtosis(self._rawTimeData, axis=1, fisher=False)
        return self._kurtosis

    def RMS(self):
        self._rms = np.sqrt(np.sum(self._rawTimeData**2, axis=1) / self._rawTimeData.shape[1])
        return self._rms
    
    def CrestFactor(self):
        self._cresfactor = self.P2P() / self.RMS()
        return self._cresfactor
    
    def ShapeFactor(self):
        self._shapefactor = self.RMS() / self.AbsMean()
        return self._shapefactor
    
    def ImpulseFactor(self):
        self._impulsefactor = self.AbsMax() / self.AbsMean()
        return self._impulsefactor

    def Features(self):
        # Time-domain Features #
        self._TimeFeatures.append(float(self.AbsMax()))        # Feature 1: Absolute Maximum 
        self._TimeFeatures.append(float(self.AbsMean()))       # Feature 2: Absolute Mean
        self._TimeFeatures.append(float(self.P2P()))           # Feature 3: Peak-to-Peak
        self._TimeFeatures.append(float(self.RMS()))           # Feature 4: Root-mean-square
        self._TimeFeatures.append(float(self.Skewness()))      # Feature 5: Skewness
        self._TimeFeatures.append(float(self.Kurtosis()))      # Feature 6: Kurtosis
        self._TimeFeatures.append(float(self.CrestFactor()))   # Feature 7: Crest Factor
        self._TimeFeatures.append(float(self.ShapeFactor()))   # Feature 8: Shape Factor
        self._TimeFeatures.append(float(self.ImpulseFactor())) # Feature 9: Impulse Factor
                
        return np.asarray(self._TimeFeatures)
    
    
    
#%%  Define class for frequency feature extraction ##
class Extract_Freq_Features:    
    def __init__(self, rawTimeData, rpm, Fs):
        self._FreqFeatures = []
        # Remove bias (subtract mean by each channel) #
        self._rawTimeData = rawTimeData - np.expand_dims(np.mean(rawTimeData,axis=1),axis=1) # Raw time data
        
        self._Fs = Fs                    # Sampling frequency [Hz]
        self._rpm = rpm/60               # RPM for every second [Hz]
       
    def FFT(self):
        # Perform FFT #
        _N = self._rawTimeData.shape[1]
        _dt = 1/self._Fs
        _yf_temp = np.fft.fft(self._rawTimeData)
        self._yf = np.abs(_yf_temp[:,:int(_N/2)]) / (_N/2)
        self._xf = np.fft.fftfreq(_N, d=_dt)[:int(_N/2)]
        
        return self._xf, self._yf
    
    def Freq_IDX(self):
        _xf,_yf = self.FFT()
        
        # Motor #
        # find frequency index of 1x #
        self._Freq_1x = self._rpm # 회전속도 
        self._1x_idx_Temp = abs(_xf - self._Freq_1x).argmin() 
        self._1x_idx_Temp = self._1x_idx_Temp - 2 + np.argmax(self._yf[0][np.arange(self._1x_idx_Temp-2, self._1x_idx_Temp+3)])
        self._1x_idx = np.arange(self._1x_idx_Temp-1, self._1x_idx_Temp+2)

        # find frequency index of 2x #
        self._Freq_2x = self._xf[self._1x_idx[1]] * 2
        self._2x_idx_Temp = abs(_xf - self._Freq_2x).argmin()
        self._2x_idx_Temp = self._2x_idx_Temp - 5 + np.argmax(self._yf[0][np.arange(self._2x_idx_Temp-5, self._2x_idx_Temp+6)])
        self._2x_idx = np.arange(self._2x_idx_Temp-1, self._2x_idx_Temp+2)
        
        # find frequency index of 3x #
        self._Freq_3x = self._xf[self._1x_idx[1]] * 3
        self._3x_idx_Temp = abs(_xf - self._Freq_3x).argmin()
        self._3x_idx_Temp = self._3x_idx_Temp - 5 + np.argmax(self._yf[0][np.arange(self._3x_idx_Temp-5, self._3x_idx_Temp+6)])
        self._3x_idx = np.arange(self._3x_idx_Temp-1, self._3x_idx_Temp+2)
        
        # find frequency index of 4x #
        self._Freq_4x = self._xf[self._1x_idx[1]]  * 4
        self._4x_idx_Temp = abs(_xf - self._Freq_4x).argmin()
        self._4x_idx_Temp = self._4x_idx_Temp - 5 + np.argmax(self._yf[0][np.arange(self._4x_idx_Temp-5, self._4x_idx_Temp+6)])
        self._4x_idx = np.arange(self._4x_idx_Temp-1, self._4x_idx_Temp+2)
        
        # Belt #
        # find frequency index of 1x #
        self._Freq_1x_B = self._Freq_1x - 5
        self._1x_idx_B_Temp = abs(_xf - self._Freq_1x_B).argmin()
        self._1x_idx_B_Temp = self._1x_idx_B_Temp - 2 + np.argmax(self._yf[0][np.arange(self._1x_idx_B_Temp-2, self._1x_idx_B_Temp+3)])
        self._1x_idx_B = np.arange(self._1x_idx_B_Temp-1, self._1x_idx_B_Temp+2)

        # find frequency index of 2x #
        self._Freq_2x_B = self._xf[self._1x_idx_B[1]] * 2
        self._2x_idx_B_Temp = abs(_xf - self._Freq_2x_B).argmin()
        self._2x_idx_B_Temp = self._2x_idx_B_Temp - 5 + np.argmax(self._yf[0][np.arange(self._2x_idx_B_Temp-5, self._2x_idx_B_Temp+6)])
        self._2x_idx_B = np.arange(self._2x_idx_B_Temp-1, self._2x_idx_B_Temp+2)
        
        # find frequency index of 3x #
        self._Freq_3x_B = self._xf[self._1x_idx_B[1]] * 3
        self._3x_idx_B_Temp = abs(_xf - self._Freq_3x_B).argmin()
        self._3x_idx_B_Temp = self._3x_idx_B_Temp - 5 + np.argmax(self._yf[0][np.arange(self._3x_idx_B_Temp-5, self._3x_idx_B_Temp+6)])
        self._3x_idx_B = np.arange(self._3x_idx_B_Temp-1, self._3x_idx_B_Temp+2)
        
        # find frequency index of 4x #
        self._Freq_4x_B = self._xf[self._1x_idx_B[1]] * 4
        self._4x_idx_B_Temp = abs(_xf - self._Freq_4x_B).argmin()
        self._4x_idx_B_Temp = self._4x_idx_B_Temp - 5 + np.argmax(self._yf[0][np.arange(self._4x_idx_B_Temp-5, self._4x_idx_B_Temp+6)])
        self._4x_idx_B = np.arange(self._4x_idx_B_Temp-1, self._4x_idx_B_Temp+2)
        
        
        return (self._1x_idx, self._2x_idx, self._3x_idx, self._4x_idx,
                self._1x_idx_B, self._2x_idx_B, self._3x_idx_B, self._4x_idx_B)

    def Features(self):
        _xf, _yf = self.FFT()
        idx = self.Freq_IDX()
                
        # Freq-domain Features #
        self._1x_Feature = np.sum(_yf[:, idx[0]], axis=1)[0]
        self._2x_Feature = np.sum(_yf[:, idx[1]], axis=1)[0]
        self._3x_Feature = np.sum(_yf[:, idx[2]], axis=1)[0] 
        self._4x_Feature = np.sum(_yf[:, idx[3]], axis=1)[0]
        self._1x_Feature_B = np.sum(_yf[:, idx[4]], axis=1)[0] 
        self._2x_Feature_B = np.sum(_yf[:, idx[5]], axis=1)[0] 
        self._3x_Feature_B = np.sum(_yf[:, idx[6]], axis=1)[0] 
        self._4x_Feature_B = np.sum(_yf[:, idx[7]], axis=1)[0] 
                
        self._FreqFeatures.append(self._1x_Feature)
        self._FreqFeatures.append(self._2x_Feature)
        self._FreqFeatures.append(self._3x_Feature)
        self._FreqFeatures.append(self._4x_Feature)
        self._FreqFeatures.append(self._1x_Feature_B)
        self._FreqFeatures.append(self._2x_Feature_B)
        self._FreqFeatures.append(self._3x_Feature_B)
        self._FreqFeatures.append(self._4x_Feature_B)
        
        return np.asarray(self._FreqFeatures)
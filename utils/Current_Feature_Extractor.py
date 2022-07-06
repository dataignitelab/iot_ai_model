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
        self._TimeFeatures.append(self.AbsMax())        # Feature 1: Absolute Maximum 
        self._TimeFeatures.append(self.AbsMean())       # Feature 2: Absolute Mean
        self._TimeFeatures.append(self.P2P())           # Feature 3: Peak-to-Peak
        self._TimeFeatures.append(self.RMS())           # Feature 4: Root-mean-square
        self._TimeFeatures.append(self.Skewness())      # Feature 5: Skewness
        self._TimeFeatures.append(self.Kurtosis())      # Feature 6: Kurtosis
        self._TimeFeatures.append(self.CrestFactor())   # Feature 7: Crest Factor
        self._TimeFeatures.append(self.ShapeFactor())   # Feature 8: Shape Factor
        self._TimeFeatures.append(self.ImpulseFactor()) # Feature 9: Impulse Factor
                
        return np.asarray(self._TimeFeatures)

    
#%%  Define class for phase feature extraction ##
class Extract_Phase_Features:    
    def __init__(self, rawTimeData, Fs):
        self._PhaseFeatures = []
        self._rawTimeData = rawTimeData - np.expand_dims(np.mean(rawTimeData,axis=1),axis=1) # Raw time data
        self._Fs = Fs                    # Sampling frequency [Hz]
        
        # Phase Shift #
    
    def Shift(self):
        
        _r = self._rawTimeData[0]
        _s = self._rawTimeData[1]
        _t = self._rawTimeData[2]
        
        _N = self._rawTimeData.shape[1]
        
        _t = np.linspace(0.0, ((_N - 1) / self._Fs), _N)
        _dts = np.linspace(-_t[-1], _t[-1], (2 * _N) - 1)
        
        # RS Phase shift
        self._RS_corr = np.correlate(_r, _s, 'full')
        self._RS_t = _dts[self._RS_corr.argmax()]
        self._RS_phase = ((2.0 * np.pi) * ((self._RS_t / (1.0 / self._Fs)) % 1.0)) - np.pi
        
        # ST Phase shift
        self._ST_corr = np.correlate(_s, _t, 'full')
        self._ST_t = _dts[self._ST_corr.argmax()]
        self._ST_phase = ((2.0 * np.pi) * ((self._ST_t / (1.0 / self._Fs)) % 1.0)) - np.pi
        
         # TR Phase shift
        self._TR_corr = np.correlate(_t, _r, 'full')
        self._TR_t = _dts[self._TR_corr.argmax()]
        self._TR_phase = ((2.0 * np.pi) * ((self._TR_t / (1.0 / self._Fs)) % 1.0)) - np.pi
        
        # Phase Level shift
        self._RS_level = _r.max() - _s.max()
        self._ST_level = _s.max() - _t.max()
        self._TR_level = _t.max() - _r.max()
        
        return (self._RS_phase, self._ST_phase, self._TR_phase, self._RS_level, self._ST_level, self._TR_level)
    
    
    def Features(self):
        shift = self.Shift()
        
        self._PhaseFeatures.append(shift[0])
        self._PhaseFeatures.append(shift[1])
        self._PhaseFeatures.append(shift[2])
        self._PhaseFeatures.append(shift[3])
        self._PhaseFeatures.append(shift[4])
        self._PhaseFeatures.append(shift[5])
        
        return np.asarray(self._PhaseFeatures)    
    
    
    
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
        self._Freq_1x = self._rpm
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
        
        
        return (self._1x_idx, self._2x_idx, self._3x_idx, self._4x_idx)

    def Features(self):
        _xf, _yf = self.FFT()
        idx = self.Freq_IDX()
                
        # Freq-domain Features #
        self._1x_Feature = np.sum(_yf[:, idx[0]], axis=1) 
        self._2x_Feature = np.sum(_yf[:, idx[1]], axis=1) 
        self._3x_Feature = np.sum(_yf[:, idx[2]], axis=1) 
        self._4x_Feature = np.sum(_yf[:, idx[3]], axis=1)
                
        self._FreqFeatures.append(self._1x_Feature)
        self._FreqFeatures.append(self._2x_Feature)
        self._FreqFeatures.append(self._3x_Feature)
        self._FreqFeatures.append(self._4x_Feature)
        
        return np.asarray(self._FreqFeatures)
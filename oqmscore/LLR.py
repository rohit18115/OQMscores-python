import numpy as np
from .OQM import OQM
from scipy.linalg import toeplitz

class LLR(OQM):
	""" Log likelihood ratio class for calculating LLR score for 
	Objective quality measure.
	
	Attributes:
		c_utt (np array of float or path to clean audio) representing the info for clean utterance
		n_utt (np array of float or path to clean audio) representing the info for noisy utterance
		c_sr (integer) sample rate for clean utterance
		n_sr (integer) sample rate for noisy utterance
		llr_score (float) stores the llr score for the speech sample
		aplha (float) do not know what this is for, please make a pull request and update it if 
		you know what it is.

	Methods: 
		score : Calculates the LLR score for the utterance.
		lpcoeff : linear predictive coefficients that help calculate the LLR
			
	"""
	def __init__(self, clean_utt='', noise_utt=''):
		
		OQM.__init__(self, clean_utt, noise_utt)
		self.llr_score = 0
		self.alpha = 0.95
	
		
	
	def score(self, eps=1e-10):
		clean_speech = self.c_utt
		processed_speech = self.n_utt
		clean_length = self.c_utt.shape[0]
		processed_length = self.n_utt.shape[0]


		assert clean_length == processed_length, clean_length

		winlength = round(30 * self.c_sr / 1000.) # 240 wlen in samples
		skiprate = np.floor(winlength / 4)
		if self.c_sr < 10000:
		    # LPC analysis order
		    P = 10
		else:
		    P = 16

		# For each frame of input speech, calculate the Log Likelihood Ratio

		num_frames = int(clean_length / skiprate - (winlength / skiprate))
		start = 0
		time = np.linspace(1, winlength, winlength) / (winlength + 1)
		window = 0.5 * (1 - np.cos(2 * np.pi * time))
		distortion = []

		for frame_count in range(num_frames):

		    # (1) Get the Frames for the test and reference speeech.
		    # Multiply by Hanning window.
		    clean_frame = clean_speech[start:start+winlength]
		    processed_frame = processed_speech[start:start+winlength]
		    clean_frame = clean_frame * window
		    processed_frame = processed_frame * window
		    # (2) Get the autocorrelation logs and LPC params used
		    # to compute the LLR measure
		    R_clean, Ref_clean, A_clean = self.lpcoeff(clean_frame, P)
		    R_processed, Ref_processed, A_processed = self.lpcoeff(processed_frame, P)
		    A_clean = A_clean[None, :]
		    A_processed = A_processed[None, :]
		    #print('A_clean shape: ', A_clean.shape)
		    #print('toe(R_clean) shape: ', toeplitz(R_clean).shape)
		    #print('A_clean: ', A_clean)
		    #print('A_processed: ', A_processed)
		    #print('toe(R_clean): ', toeplitz(R_clean))
		    # (3) Compute the LLR measure
		    numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
		    #print('num_1: {}'.format(A_processed.dot(toeplitz(R_clean))))
		    #print('num: ', numerator)
		    denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)
		    #print('den: ', denominator)
		    #log_ = np.log(max(numerator / denominator, 10e-20))
		    #print('R_clean: ', R_clean)
		    #print('num: ', numerator)
		    #print('den: ', denominator)
		    #raise NotImplementedError
		    log_ = np.log(numerator / denominator)
		    #print('np.log({}/{}) = {}'.format(numerator, denominator, log_))
		    distortion.append(np.squeeze(log_))
		    start += int(skiprate)

		LLR_dist = np.array(distortion)
		LLR_dist = sorted(LLR_dist, reverse=False)
		LLRs = LLR_dist
		LLR_len = round(len(LLR_dist) * self.alpha)
		llr_mean = np.mean(LLRs[:LLR_len])
		self.llr_score = llr_mean

	def lpcoeff(self, speech_frame, model_order):
    
		# (1) Compute Autocor lags
		# max?
		winlength = speech_frame.shape[0]
		R = []
		#R = [0] * (model_order + 1)
		for k in range(model_order + 1):
		    first = speech_frame[:(winlength - k)]
		    second = speech_frame[k:winlength]
		    #raise NotImplementedError
		    R.append(np.sum(first * second))
		    #R[k] = np.sum( first * second)
		# (2) Lev-Durbin
		a = np.ones((model_order,))
		E = np.zeros((model_order + 1,))
		rcoeff = np.zeros((model_order,))
		E[0] = R[0]
		for i in range(model_order):
		    #print('-' * 40)
		    #print('i: ', i)
		    if i == 0:
		        sum_term = 0
		    else:
		        a_past = a[:i]
		        #print('R[i:0:-1] = ', R[i:0:-1])
		        #print('a_past = ', a_past)
		        sum_term = np.sum(a_past * np.array(R[i:0:-1]))
		        #print('a_past size: ', a_past.shape)
		    #print('sum_term = {:.6f}'.format(sum_term))
		    #print('E[i] =  {}'.format(E[i]))
		    #print('R[i+1] = ', R[i+1])
		    rcoeff[i] = (R[i+1] - sum_term)/E[i]
		    #print('len(a) = ', len(a))
		    #print('len(rcoeff) = ', len(rcoeff))
		    #print('a[{}]={}'.format(i, a[i]))
		    #print('rcoeff[{}]={}'.format(i, rcoeff[i]))
		    a[i] = rcoeff[i]
		    if i > 0:
		        #print('a: ', a)
		        #print('a_past: ', a_past)
		        #print('a_past[:i] ', a_past[:i])
		        #print('a_past[::-1] ', a_past[::-1])
		        a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
		    E[i+1] = (1-rcoeff[i]*rcoeff[i])*E[i]
		    #print('E[i+1]= ', E[i+1])
		acorr = np.array(R, dtype=np.float32)
		refcoeff = np.array(rcoeff, dtype=np.float32)
		a = a * -1
		lpparams = np.array([1] + list(a), dtype=np.float32)
		acorr =np.array(acorr, dtype=np.float32)
		refcoeff = np.array(refcoeff, dtype=np.float32)
		lpparams = np.array(lpparams, dtype=np.float32)
		#print('acorr shape: ', acorr.shape)
		#print('refcoeff shape: ', refcoeff.shape)
		#print('lpparams shape: ', lpparams.shape)
		return acorr, refcoeff, lpparams

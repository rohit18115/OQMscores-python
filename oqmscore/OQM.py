import librosa as lb
import numpy as np


class OQM:


	def __init__(self, clean_utt = '', noise_utt = ''):
	
		""" Generic Objective quality measure class for initializing variables and loading 
		the speech samples
	
		Attributes:
			c_utt (np array of float or path to clean audio) representing the info for clean utterance
			n_utt (np array of float or path to clean audio) representing the info for noisy utterance
			c_sr (integer) sample rate for clean utterance
			n_sr (integer) sample rate for noisy utterance
			"""
		
		self.c_utt = clean_utt
		self.n_utt = noise_utt
		self.c_sr = 0
		self.n_sr = 0



	def load(self, clean_utt,noise_utt):
	
		"""Function to load the clean and noisy utterance if the path is specified.
		OR if numpy arrays are passed to the function then it just resamples it with 
		sample rate equal to 16000(its a requirement for the PESQ class to work properly)
				
		Args:
			file_name (string): name of a file to read from
		
		Returns:
			None
		
		"""
			
		if isinstance(clean_utt, str) and isinstance(noise_utt, str):
			self.c_utt, self.c_sr = lb.load(clean_utt, sr=16000)
			self.n_utt, self.n_sr = lb.load(noise_utt, sr=16000)
			self.c_utt = np.array(self.c_utt).reshape(-1)
			self.n_utt = np.array(self.n_utt).reshape(-1)

		else:

			self.c_utt = np.array(clean_utt).reshape(-1)
			self.n_utt = np.array(noise_utt).reshape(-1)
			self.c_utt = lb.resample(self.c_utt, self.c_sr,16000)
			self.c_utt = lb.resample(self.n_utt, self.n_sr,16000)


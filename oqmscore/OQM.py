import librosa as lb
import numpy as np


class OQM:


	def __init__(self, clean_utt = '', noise_utt = ''):
	
		""" Generic distribution class for calculating and 
		visualizing a probability distribution.
	
		Attributes:
			mean (float) representing the mean value of the distribution
			stdev (float) representing the standard deviation of the distribution
			data_list (list of floats) a list of floats extracted from the data file
			"""
		
		self.c_utt = clean_utt
		self.n_utt = noise_utt
		self.c_sr = 0
		self.n_sr = 0



	def load(self, clean_utt,noise_utt):
	
		"""Function to read in data from a txt file. The txt file should have
		one number (float) per line. The numbers are stored in the data attribute.
				
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


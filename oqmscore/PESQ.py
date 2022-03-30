import numpy as np
from pesq import pesq
from .OQM import OQM

class PESQ(OQM):
	""" Gaussian distribution class for calculating and 
	visualizing a Gaussian distribution.
	
	Attributes:
		mean (float) representing the mean value of the distribution
		stdev (float) representing the standard deviation of the distribution
		data_list (list of floats) a list of floats extracted from the data file
			
	"""
	def __init__(self, clean_utt='', noise_utt=''):
		
		OQM.__init__(self, clean_utt, noise_utt)
		self.pesq_score = 0
	
		
	
	def score(self):
		self.pesq_score = pesq(ref=self.c_utt, deg=self.n_utt, fs=self.c_sr, mode='wb')
		if self.pesq_score is not None:
			self.pesq_score = float(self.pesq_score)
		else:
			self.pesq_score = -1.

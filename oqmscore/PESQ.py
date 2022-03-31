import numpy as np
from pesq import pesq
from .OQM import OQM

class PESQ(OQM):
	""" PESQ class for calculating PESQ score for 
	Objective quality measure.
	
	Attributes:
		c_utt (np array of float or path to clean audio) representing the info for clean utterance
		n_utt (np array of float or path to clean audio) representing the info for noisy utterance
		c_sr (integer) sample rate for clean utterance
		n_sr (integer) sample rate for noisy utterance
		pesq_score (float) stores the PESQ score for the speech sample
		

	Methods: 
		score : Calculates the PESQ score for the utterance.
		

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

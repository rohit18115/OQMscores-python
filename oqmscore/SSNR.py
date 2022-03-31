import numpy as np
from .OQM import OQM

class SSNR(OQM):
	""" Segmental signal to noise ratio class for calculating SSNR score for 
	Objective quality measure.
	
	Attributes:
		c_utt (np array of float or path to clean audio) representing the info for clean utterance
		n_utt (np array of float or path to clean audio) representing the info for noisy utterance
		c_sr (integer) sample rate for clean utterance
		n_sr (integer) sample rate for noisy utterance
		ssnr_score (float) stores the ssnr score for the speech sample
		

	Methods: 
		score : Calculates the ssnr score for the utterance.
			
	"""
	def __init__(self, clean_utt='', noise_utt=''):
		
		OQM.__init__(self, clean_utt, noise_utt)
		self.ssnr_score = 0
	
		
	
	def score(self, eps=1e-10):
		clean_speech = self.c_utt
		processed_speech = self.n_utt
		clean_length = self.c_utt.shape[0]
		processed_length = self.n_utt.shape[0]


		# scale both to have same dynamic range. Remove DC too.
		dif = self.c_utt - self.n_utt
		overall_snr = 10 * np.log10(np.sum(self.c_utt ** 2) / (np.sum(dif ** 2) +
		                                                    10e-20))

		# global variables
		winlength = int(np.round(30 * self.c_sr / 1000)) # 30 msecs
		skiprate = winlength // 4
		MIN_SNR = -10
		MAX_SNR = 35

		# For each frame, calculate SSNR

		num_frames = int(clean_length / skiprate - (winlength/skiprate))
		start = 0
		time = np.linspace(1, winlength, winlength) / (winlength + 1)
		window = 0.5 * (1 - np.cos(2 * np.pi * time))
		segmental_snr = []

		for frame_count in range(int(num_frames)):
		    # (1) get the frames for the test and ref speech.
		    # Apply Hanning Window
		    clean_frame = clean_speech[start:start+winlength]
		    processed_frame = processed_speech[start:start+winlength]
		    clean_frame = clean_frame * window
		    processed_frame = processed_frame * window

		    # (2) Compute Segmental SNR
		    signal_energy = np.sum(clean_frame ** 2)
		    noise_energy = np.sum((clean_frame - processed_frame) ** 2)
		    segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps)+ eps))
		    segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
		    segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
		    start += int(skiprate)
		self.ssnr_score = np.mean(segmental_snr)

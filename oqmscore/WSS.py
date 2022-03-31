import numpy as np
from .OQM import OQM

class WSS(OQM):
	""" Weighted spectral slope class for calculating WSS score for 
	Objective quality measure.
	
	Attributes:
		c_utt (np array of float or path to clean audio) representing the info for clean utterance
		n_utt (np array of float or path to clean audio) representing the info for noisy utterance
		c_sr (integer) sample rate for clean utterance
		n_sr (integer) sample rate for noisy utterance
		wss_score (float) stores the ssnr score for the speech sample
		aplha (float) do not know what this is for, please make a pull request and update it if 
		you know what it is.
		

	Methods: 
		score : Calculates the wss score for the utterance.
			
	"""
	def __init__(self, clean_utt='', noise_utt=''):
		
		OQM.__init__(self, clean_utt, noise_utt)
		self.wss_score = 0
		self.alpha = 0.95
	
		
	
	def score(self, eps=1e-10):
		clean_speech = self.c_utt
		processed_speech = self.n_utt
		clean_length = self.c_utt.shape[0]
		processed_length = self.n_utt.shape[0]


		assert clean_length == processed_length, clean_length

		winlength = round(30 * self.c_sr / 1000.) # 240 wlen in samples
		skiprate = np.floor(winlength / 4)
		max_freq = self.c_sr / 2
		num_crit = 25 # num of critical bands

		USE_FFT_SPECTRUM = 1
		n_fft = int(2 ** np.ceil(np.log(2*winlength)/np.log(2)))
		n_fftby2 = int(n_fft / 2)
		Kmax = 20
		Klocmax = 1

		# Critical band filter definitions (Center frequency and BW in Hz)

		cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
		             703.378, 798.717, 904.128, 1020.38, 1148.30, 
		             1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 
		             2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
		             3597.63]
		bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
		             95.3398, 105.411, 116.256, 127.914, 140.423, 
		             153.823, 168.154, 183.457, 199.776, 217.153, 
		             235.631, 255.255, 276.072, 298.126, 321.465,
		             346.136]

		bw_min = bandwidth[0] # min critical bandwidth

		# set up critical band filters. Note here that Gaussianly shaped filters
		# are used. Also, the sum of the filter weights are equivalent for each
		# critical band filter. Filter less than -30 dB and set to zero.

		min_factor = np.exp(-30. / (2 * 2.303)) # -30 dB point of filter

		crit_filter = np.zeros((num_crit, n_fftby2))
		all_f0 = []
		for i in range(num_crit):
		    f0 = (cent_freq[i] / max_freq) * (n_fftby2)
		    all_f0.append(np.floor(f0))
		    bw = (bandwidth[i] / max_freq) * (n_fftby2)
		    norm_factor = np.log(bw_min) - np.log(bandwidth[i])
		    j = list(range(n_fftby2))
		    crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + \
		                               norm_factor)
		    crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > \
		                                             min_factor)
		# For each frame of input speech, compute Weighted Spectral Slope Measure

		# num of frames
		num_frames = int(clean_length / skiprate - (winlength / skiprate))
		start = 0 # starting sample
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

		    # (2) Compuet Power Spectrum of clean and processed

		    clean_spec = (np.abs(np.fft.fft(clean_frame, n_fft)) ** 2)
		    processed_spec = (np.abs(np.fft.fft(processed_frame, n_fft)) ** 2)
		    clean_energy = [None] * num_crit
		    processed_energy = [None] * num_crit
		    # (3) Compute Filterbank output energies (in dB)
		    for i in range(num_crit):
		        clean_energy[i] = np.sum(clean_spec[:n_fftby2] * \
		                                 crit_filter[i, :])
		        processed_energy[i] = np.sum(processed_spec[:n_fftby2] * \
		                                     crit_filter[i, :])
		    clean_energy = np.array(clean_energy).reshape(-1, 1)
		    eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
		    clean_energy = np.concatenate((clean_energy, eps), axis=1)
		    clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
		    processed_energy = np.array(processed_energy).reshape(-1, 1)
		    processed_energy = np.concatenate((processed_energy, eps), axis=1)
		    processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))
		    # (4) Compute Spectral Shape (dB[i+1] - dB[i])

		    clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit-1]
		    processed_slope = processed_energy[1:num_crit] - \
		            processed_energy[:num_crit-1]
		    # (5) Find the nearest peak locations in the spectra to each
		    # critical band. If the slope is negative, we search
		    # to the left. If positive, we search to the right.
		    clean_loc_peak = []
		    processed_loc_peak = []
		    for i in range(num_crit - 1):
		        if clean_slope[i] > 0:
		            # search to the right
		            n = i
		            while n < num_crit - 1 and clean_slope[n] > 0:
		                n += 1
		            clean_loc_peak.append(clean_energy[n - 1])
		        else:
		            # search to the left
		            n = i
		            while n >= 0 and clean_slope[n] <= 0:
		                n -= 1
		            clean_loc_peak.append(clean_energy[n + 1])
		        # find the peaks in the processed speech signal
		        if processed_slope[i] > 0:
		            n = i
		            while n < num_crit - 1 and processed_slope[n] > 0:
		                n += 1
		            processed_loc_peak.append(processed_energy[n - 1])
		        else:
		            n = i
		            while n >= 0 and processed_slope[n] <= 0:
		                n -= 1
		            processed_loc_peak.append(processed_energy[n + 1])
		    # (6) Compuet the WSS Measure for this frame. This includes
		    # determination of the weighting functino
		    dBMax_clean = max(clean_energy)
		    dBMax_processed = max(processed_energy)
		    # The weights are calculated by averaging individual
		    # weighting factors from the clean and processed frame.
		    # These weights W_clean and W_processed should range
		    # from 0 to 1 and place more emphasis on spectral 
		    # peaks and less emphasis on slope differences in spectral
		    # valleys.  This procedure is described on page 1280 of
		    # Klatt's 1982 ICASSP paper.
		    clean_loc_peak = np.array(clean_loc_peak)
		    processed_loc_peak = np.array(processed_loc_peak)
		    Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit-1])
		    Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - \
		                               clean_energy[:num_crit-1])
		    W_clean = Wmax_clean * Wlocmax_clean
		    Wmax_processed = Kmax / (Kmax + dBMax_processed - \
		                            processed_energy[:num_crit-1])
		    Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - \
		                                  processed_energy[:num_crit-1])
		    W_processed = Wmax_processed * Wlocmax_processed
		    W = (W_clean + W_processed) / 2
		    distortion.append(np.sum(W * (clean_slope[:num_crit - 1] - \
		                                 processed_slope[:num_crit - 1]) ** 2))

		    # this normalization is not part of Klatt's paper, but helps
		    # to normalize the meaasure. Here we scale the measure by the sum of the
		    # weights
		    distortion[frame_count] = distortion[frame_count] / np.sum(W)
		    start += int(skiprate)
		wss_dist_vec = distortion
		wss_dist_vec = sorted(wss_dist_vec, reverse=False)
		wss_dist = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * self.alpha))])
		self.wss_score = wss_dist

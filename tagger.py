import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	state = np.zeros(len(tags), dtype="int")
	pi = np.zeros(len(state))
	state_dict = {}
	obs_dict = {}
	word_set = set()

	for i, tag in enumerate(tags):
		state_dict[tag] = i

	for sentence in train_data:
		state[state_dict[sentence.tags[0]]] += 1
	pi = np.divide(state, sum(state))

	for sentence in train_data:
		for word in sentence.words:
			word_set.add(word)

	obs_symbols = list(word_set)
	for i, symbol in enumerate(obs_symbols):
		obs_dict[symbol] = i

	A = np.zeros([len(state), len(state)])
	for sentence in train_data:
		for i in range(len(sentence.tags) - 1):
			A[state_dict[sentence.tags[i]]][state_dict[sentence.tags[i + 1]]] += 1

	sumOfRow = np.sum(A, axis = 1)
	A = np.divide(A, sumOfRow.reshape(-1, 1))

	B = np.zeros([len(state), len(word_set)])
	for sentence in train_data:
		for tag, word in zip(sentence.tags, sentence.words):
			B[state_dict[tag]][obs_dict[word]] += 1
	sumOfRow = np.sum(B, axis = 1)
	B = np.divide(B, sumOfRow.reshape(-1, 1))
	model = HMM(pi, A, B, obs_dict, state_dict)

	# print("Values (", "pi : ", pi, "\nA : ", A, "\nB : ", B, "\nobs dict : ", obs_dict, "\nstate dict : ", state_dict)
	###################################################
	"""
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
	"""

	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	for sentence in test_data:
		for word in sentence.words:
			if not word in model.obs_dict:
				b = np.asarray([10 ** -6] * len(model.pi)).reshape(-1, 1)
				model.B = np.column_stack((model.B, b))
				model.obs_dict[word] = len(model.obs_dict)

		path = model.viterbi(sentence.words)
		tagging.append(path)
	###################################################

	return tagging

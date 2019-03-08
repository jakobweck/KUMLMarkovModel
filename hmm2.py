# A class for performing hidden markov models

import copy
import numpy as np
import sys
import re

class HMM():

    def __init__(self, transmission_prob_beta, emission_prob, obs=None):
        '''
        Note that this implementation assumes that n, m, and T are small
        enough not to require underflow mitigation.

        Required Inputs:
        - transmission_prob_beta: an (n+2) x (n+2) numpy array, initial, where n is
        the number of hidden states
        - emission_prob: an (m x n) 2-D numpy array, where m is the number of
        possible observations

        Optional Input:
        - obs: a list of observation labels, in the same order as their
        occurence within the emission probability matrix; otherwise, will assume
        that the emission probabilities are in alpha-numerical order.
        '''
        #pi_state is transmission_prob_beta[state][0] - probability of starting in state at the beginning
        self.transmission_prob_beta = transmission_prob_beta
        #emission_prob[emiss_ref[observation]][state] - probability of getting [observation] in [state]
        self.emission_prob = emission_prob
        self.n = self.emission_prob.shape[1]
        self.m = self.emission_prob.shape[0]
        self.observations = None
        self.forward = []
        self.forwardnorm = []
        self.backward = []
        self.backwardnorm[]
        self.psi = []
        self.obs = obs
        self.emiss_ref = {}
        self.forward_final = [0 , 0]
        self.backward_final = [0 , 0]
        self.state_probs = []
        if obs is None and self.observations is not None:
            self.obs = self.assume_obs()

    def assume_obs(self):
        '''
        If observation labels are not given, will assume that the emission
        probabilities are in alpha-numerical order.
        '''
        obs = list(set(list(self.observations)))
        obs.sort()
        for i in range(len(obs)):
            self.emiss_ref[obs[i]] = i
        return obs

    def train(self, observations, iterations = 10, verbose=True):
        '''
        Trains the model parameters according to the observation sequence.

        Input:
        - observations: 1-D string array of T observations
        '''
        breakpoint()
        self.observations = observations
        self.obs = self.assume_obs()
        #list of zeroes for each state, for each state
        self.psi = [[[0.0] * (len(self.observations)-1) for i in range(self.n)] for i in range(self.n)]
        #list of zeroes for each state
        self.gamma = [[0.0] * (len(self.observations)) for i in range(self.n)]
        for i in range(iterations):
            old_transmission = self.transmission_prob_beta.copy()
            old_emission = self.emission_prob.copy()
            if verbose:
                print("Iteration: {}".format(i + 1))
            self.expectation()
            self.maximization()

    def expectation(self):
        '''
        Executes expectation step.
        '''
        self.forward = self.forward_recurse(len(self.observations))
        self.forwardnorm = self.forward_recurse_norm(len(self.observations))
        self.backward = self.backward_recurse(0)
        self.backward = self.backward_recurse_norm(0)
        self.get_gamma()
        self.get_psi()

    def get_gamma(self):
        '''
        Calculates the gamma matrix.
        '''
        self.gamma = [[0, 0] for i in range(len(self.observations))]
        for i in range(len(self.observations)):
            self.gamma[i][0] = (float(self.forward[0][i] * self.backward[0][i]) /
                                float(self.forward[0][i] * self.backward[0][i] +
                                self.forward[1][i] * self.backward[1][i]))
            self.gamma[i][1] = (float(self.forward[1][i] * self.backward[1][i]) /
                                float(self.forward[0][i] * self.backward[0][i] +
                                self.forward[1][i] * self.backward[1][i]))

    def get_psi(self):
        '''
        Runs the psi calculation.
        '''
        for t in range(1, len(self.observations)):
            for j in range(self.n):
                for i in range(self.n):
                    self.psi[i][j][t-1] = self.calculate_psi(t, i, j)

    def calculate_psi(self, t, i, j):
        '''
        Calculates the psi for a transition from i->j for t > 0.
        '''
        alpha_tminus1_i = self.forward[i][t-1]
        a_i_j = self.transmission_prob_beta[j+1][i+1]
        beta_t_j = self.backward[j][t]
        observation = self.observations[t]
        b_j = self.emission_prob[self.emiss_ref[observation]][j]
        denom = float(self.forward[0][i] * self.backward[0][i] + self.forward[1][i] * self.backward[1][i])
        return (alpha_tminus1_i * a_i_j * beta_t_j * b_j) / denom

    def maximization(self):
        '''
        Executes maximization step.
        '''
        self.get_state_probs()
        for i in range(self.n):
            self.transmission_prob_beta[i+1][0] = self.gamma[0][i]
            self.transmission_prob_beta[-1][i+1] = self.gamma[-1][i] / self.state_probs[i]
            for j in range(self.n):
                self.transmission_prob_beta[j+1][i+1] = self.estimate_transmission(i, j)
            for obs in range(self.m):
                self.emission_prob[obs][i] = self.estimate_emission(i, obs)

    def get_state_probs(self):
        '''
        Calculates total probability of a given state.
        '''
        self.state_probs = [0] * self.n
        for state in range(self.n):
            summ = 0
            for row in self.gamma:
                summ += row[state]
            self.state_probs[state] = summ

    def estimate_transmission(self, i, j):
        '''
        Estimates transmission probabilities from i to j.
        '''
        return sum(self.psi[i][j]) / self.state_probs[i]

    def estimate_emission(self, j, observation):
        '''
        Estimate emission probability for an observation from state j.
        '''
        observation = self.obs[observation]
        ts = [i for i in range(len(self.observations)) if self.observations[i] == observation]
        for i in range(len(ts)):
            ts[i] = self.gamma[ts[i]][j]
        return sum(ts) / self.state_probs[j]

    def backward_recurse(self, index):
        '''
        Runs the backward recursion.
        '''
        # Initialization at T
        if index == (len(self.observations) - 1):
            backward = [[0.0] * (len(self.observations)) for i in range(self.n)]
            for state in range(self.n):
                backward[state][index] = self.backward_initial(state)
            return backward
        # Recursion for T --> 0
        else:
            backward = self.backward_recurse(index+1)
            for state in range(self.n):
                if index >= 0:
                    backward[state][index] = self.backward_probability(index, backward, state)
                if index == 0:
                    self.backward_final[state] = self.backward_probability(index, backward, 0, final=True)
            return backward

    def backward_initial(self, state):
        '''
        Initialization of backward probabilities.
        '''
        return self.transmission_prob_beta[self.n + 1][state + 1]

    def backward_probability(self, index, backward, state, final=False):
        '''
        Calculates the backward probability at index = t.
        '''
        p = [0] * self.n
        for j in range(self.n):
            observation = self.observations[index + 1]
            if not final:
                a = self.transmission_prob_beta[j + 1][state + 1]
            else:
                a = self.transmission_prob_beta[j + 1][0]
            b = self.emission_prob[self.emiss_ref[observation]][j]
            beta = backward[j][index + 1]
            p[j] = a * b * beta
        return sum(p)

    def forward_recurse(self, index):
        '''
        Executes forward recursion.
        '''
        # Initialization - calculate ai(0)
        if index == 0:
            forward = [[0.0] * (len(self.observations)) for i in range(self.n)]
            for state in range(self.n):
                forward[state][index] = self.forward_initial(self.observations[index], state)
            return forward
        # Recursion
        else:
            forward = self.forward_recurse(index-1)
            for state in range(self.n):
                if index != len(self.observations):
                    forward[state][index] = self.forward_probability(index, forward, state)
                else:
                    # Termination
                    self.forward_final[state] = self.forward_probability(index, forward, state, final=True)
            return forward

    def forward_initial_norm(self, observation, state):
        '''
        Calculates initial forward probabilities.
        '''
        self.transmission_prob_beta[state + 1][0]
        self.emission_prob[self.emiss_ref[observation]][state]
        breakpoint()
        top = self.transmission_prob_beta[state + 1][0] * self.emission_prob[self.emiss_ref[observation]][state]
        div = 0.0
        #normalize to avoid underflow
        for k in self.transmission_prob_beta:
            div += self.transmission_prob_beta[k][0] * self.emission_prob[self.emiss_ref[observation]][state]
        return top/div

    def forward_initial(self, observation, state):
        '''
        Calculates initial forward probabilities.
        '''
        self.transmission_prob_beta[state + 1][0]
        self.emission_prob[self.emiss_ref[observation]][state]
        breakpoint()
        res = self.transmission_prob_beta[state + 1][0] * self.emission_prob[self.emiss_ref[observation]][state]
        return res

    def forward_probability(self, index, forward, state, final=False):
        '''
        Calculates the alpha for index = t.
        '''
        p = [0] * self.n
        for prev_state in range(self.n):
            if not final:
                # Recursion
                obs_index = self.emiss_ref[self.observations[index]]
                p[prev_state] = forward[prev_state][index-1] * self.transmission_prob_beta[state + 1][prev_state + 1] * self.emission_prob[obs_index][state]
            else:
                # Termination
                p[prev_state] = forward[prev_state][index-1] * self.transmission_prob_beta[self.n][prev_state + 1]
        return sum(p)

    def forward_probability_norm(self, index, forward, state, final=False):
        '''
        Calculates the alpha for index = t.
        '''
        p = [0] * self.n
        divLeft = 0.0
        for prev_state in range(self.n):
            if not final:
                # Recursion
                obs_index = self.emiss_ref[self.observations[index]]
                p[prev_state] = forward[prev_state][index-1] * self.transmission_prob_beta[state + 1][prev_state + 1] * self.emission_prob[obs_index][state]
            else:
                # Termination
                p[prev_state] = forward[prev_state][index-1] * self.transmission_prob_beta[self.n][prev_state + 1]
        #normalize
        for k in range(self.n):
            divRight = 0.0
            divLeft += self.transmission_prob_beta[state+1][k]
            for j in range(self.n):
                divRight = forward[][]*self.forwardnorm[][]
        return sum(p) /= (divLeft)

    def likelihood(self, new_observations):
        '''
        Returns the probability of a observation sequence based on current model
        parameters.
        '''
        new_hmm = HMM(self.transmission_prob_beta, self.emission_prob)
        new_hmm.observations = new_observations
        new_hmm.obs = new_hmm.assume_obs()
        forward = new_hmm.forward_recurse(len(new_observations))
        return sum(new_hmm.forward_final)

def main():
    # Example inputs from Jason Eisner's Ice Cream and Baltimore Summer example
    # http://www.cs.jhu.edu/~jason/papers/#eisner-2002-tnlp
    emission = np.array([[0.7, 0], [0.2, 0.3], [0.1, 0.7]])
    transmission = np.array([ [0, 0, 0, 0], [0.5, 0.8, 0.2, 0], [0.5, 0.1, 0.7, 0], [0, 0.1, 0.1, 0]])
    observations = ['2','3','3','2','3','2','3','2','2','3','1','3','3','1','1',
                    '1','2','1','1','1','3','1','2','1','1','1','2','3','3','2',
                    '3','2','2']
    inputFileName = sys.argv[1]
    words = []
    with open(inputFileName, 'r') as myfile:
        data=myfile.read().lower()
        #parse file for words only
        words = np.array(re.findall(r"[\w']+|[.,!?;]", data))[:500]
    #num hidden states let's say 2
    #num possible observations is number of unique words in words
    emission = np.random.random(2)
    emission /= emission.sum()
    m = len(set(words))
    print(m)
    for i in range(m-1):
        arr = np.random.random(2) 
        arr /= arr.sum() #random emission probs from the 2 states that add up to 1 for each possible word
        emission = np.vstack([emission, arr])
    model = HMM(transmission, emission)
    model.train(words)
    print("Model transmission probabilities:\n{}".format(model.transmission_prob_beta))
    print("Model emission probabilities:\n{}".format(model.emission_prob))
    # Probability of a new sequence
    new_seq = ['beastly', 'shameless', 'transformation']
    print("Finding likelihood for {}".format(new_seq))
    likelihood = model.likelihood(new_seq)
    print("Likelihood: {}".format(likelihood))


main()
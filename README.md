# Hidden Markov Model
## Austin Irvine and Jakob Wulf-Eck

Requires pandas, numpy, and scipy (for log-probability operations). 
Takes as input a .txt corpus of raw, natural language text. Use hmmtrain.py -(filename) -(number of hidden states) -(number of words to take from corpus) to train a hidden Markov model with the specified number of states on the corpus. This (lengthy) operation will ultimately output emit_final, trans_final, and prior csv files representing the trained model. Run genfromtrained.py to generate or predict text based on the trained model. Training was generally tested with ~5-8 states and ~10000 input words - training on larger datasets may take a very long time.

We based our approach to training the model on Emilio Esposito's implementation (https://github.com/EmilioEsposito/hmm-hidden-markov-model). This approach uses log-probability for all probability calculations, preventing the underflow that results from iterative arithmetic operations on very small probabilities. Using the corpus text as our observation sequence, we iteratively perform the Baum-Welch forward-backward algorithm on a random set of input parameters generated using the Dirichlet distribution. We continue iteration until the change in the forward probability matrix between iterations passes below a convergence threshold. Because of the resource-intense nature of the algorithm over our dataset, we relax the convergence requirements as the iteration count increases.

The particular challenge of applying hidden Markov models to natural language is the massive set of possible observations - loading even a tenth of the Shakespare dataset into the training script results in thousands of possible symbols which can be emitted, drastically extending the training process. Unlike applications such as speech recognition, there is little literature available which speculates on the ideal number of hidden states to represent language output - although 7 or 8 state models seem to converge most quickly in training. Few reference HMM implementations are available which account for the underflow mitigation and performance requirements of training on input with so many possible observations - many canonical examples operate on less than five output 'words'.

In addition to the hidden Markov model, standardmarkov.py -(filename) can be used to generate text using a standard, transition-matrix based Markov model (without hidden states) with 1-word prefix size.
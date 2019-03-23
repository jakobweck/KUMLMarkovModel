# Hidden Markov Model
## Austin Irvine and Jakob Wulf-Eck

Requires pandas, numpy, and scipy (for log-probability operations). Use hmmtrain.py -(filename) -(number of hidden states) to train the model. This produces emit_final, trans_final, and prior csv files representing the trained model. Run genfromtrained.py to generate or predict text based on the trained model.
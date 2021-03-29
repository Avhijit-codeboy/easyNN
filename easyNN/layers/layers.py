import pandas as pd
import numpy as np
def layers(input_file,no_of_units_in_each_hidden_layer,output_units):
	df = pd.read_csv(input_file) #currently only csv file supported
	input_neurons = df.shape[0] # number of input units
	layer_list = [input_neurons] # list of neuron information for different layer includinh input and output layers

	for i in no_of_units_in_each_hidden_layer:
		layer_list.append(i)
	layer_list.append(output_units)

	weight_matrix = [] #list of weight matrices

	for i in range(len(layer_list)-1):
		a = np.zeros((layer_list[i+1],layer_list[i])) # dimensions of weight matrices
		weight_matrix.append(a) # appending each weight matrix to a single list

	return weight_matrix

#ans = layers("sloan_sky_survey.csv",[4,4],2)
#print(ans[0].shape," ",ans[1].shape," ",ans[2].shape)

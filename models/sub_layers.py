import torch
import torch.nn as nn
import pdb 
class mlp_layers(nn.Module):
	def __init__(self,input_dims,num_layer,dropout=0.1, internal_dims=None,output_dims = None,conv_mode = None):
		super(mlp_layers,self).__init__()
		self.input_dims = input_dims
		self.dropout = dropout
		layer_list = []
		
		if internal_dims is None:
			internal_dims = input_dims
		if output_dims is None:
			output_dims = input_dims
		for i in range(num_layer):
			if i == num_layer-1:
				internal_dims = output_dims	
			layer_list.append(nn.Linear(input_dims,internal_dims))
			layer_list.append(nn.ReLU())
			if dropout >0 :
				layer_list.append(nn.Dropout(p=dropout))
			input_dims = internal_dims
		self.mlp_layer = nn.Sequential(*layer_list)
	def forward(self,x):
		return (self.mlp_layer(x))

class Highway_layer(nn.Module):
	def __init__(self,input_size,activation='ReLU'):

		super(Highway_layer,self).__init__()
		self.transform_gate = nn.Sequential(nn.Linear(input_size,input_size),nn.Sigmoid())
		activation_layer = getattr(nn,activation)()
		self.h_layer = nn.Sequential(nn.Linear(input_size,input_size),activation_layer)
		#init_model
	def init_model(self):
		raise NotImplementedError
		
	def forward(self,x):
		tf_gate_out = self.transform_gate(x)
		out = self.h_layer(x)*(tf_gate_out) + (1-tf_gate_out)*x
		return out
		
import torch
import torch.nn 

class mlp_layers(nn.Module):
	def __init__(self,input_dims,num_layer,dropout=0.1, internal_dims=None,output_dims = None,conv_mode = None):
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
			layer_list = internal_dims
		self.mlp_layer = nn.Sequential(*layer_list)
	def forward(self,x):
		return (self.mlp_layer(x))
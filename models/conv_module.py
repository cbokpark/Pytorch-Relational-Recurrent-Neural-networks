import torch
import torch.nn as nn 
from .sub_layers import Highway_layer
import pdb 

class CharEncodeNework(nn.Module):
	def __init__(self,num_vocab,max_length,n_highway,char_embedd_size,conv_options,dropout_p=0.5,project_dim = None,use_wordvec =False,word_vec_size =(1,1),use_bn =True):
		super(CharEncodeNework,self).__init__()
		self.num_vocab = num_vocab
		self.max_length = max_length
		self.conv_options = conv_options
		self.use_wordvec = use_wordvec
		self.char_vec = nn.Embedding(num_vocab,char_embedd_size)
		self.char_conv_layer = char_conv_embedding(conv_options,char_embedd_size,max_length)
		conv_output_size = sum(conv_options.values())
		if project_dim is not None:
			self.project_dim = project_dim
			self.project_layer = nn.Linear(conv_output_size,project_dim) 
			self.final_output_size = project_dim
		else:
			self.project_layer = None
			self.final_output_size = conv_output_size
		self.conv_output_size = conv_output_size
		
		if use_wordvec:
			self.word_vec = nn.Embedding(word_vec_size(0),word_vec_size(1))
			conv_output_size += word_vec_size(1)

		if use_bn:
			self.batch_norm = nn.BatchNorm1d(conv_output_size)
	
		self.n_highway = n_highway
		self.dropout_p = nn.Dropout(dropout_p)
		if n_highway > 0:	
			highway_layer = [Highway_layer(input_size = conv_output_size,activation ='ReLU') for i in range(n_highway)]
			self.highway_layer = nn.Sequential(*highway_layer)


	def forward(self,x,word_index = None):
		"""
			Params :
			 x : Batch x Time_sequence x max_length 

			Returns:
			 outputs : Batch x Time_sequence x d_model 
		"""
		batch_size ,time_step,_ = x.size()
		x = self.char_vec(x)

		x = self.char_conv_layer(x)

		if self.use_wordvec:
			wordvec = self.word_vec(word_index)
			x = torch.cat((x,word_index),dim=-1)
		if self.batch_norm:
			x = self.batch_norm(x)
		x = self.dropout_p(x)

		output = self.highway_layer(x)
		if self.project_layer:
			output = self.project_layer(output)
		output = output.view(batch_size,time_step,-1)

		return output 






class char_conv_embedding(nn.Module):
	"""
		Convolution layer with multiple filters of different widths 
	"""
	def __init__(self,conv_options,char_embedd,max_length,in_channel=1):
		"""
		Prams:
			conv_options : dict;{ filter_size : filter_num}
			embedding_size: charcter_embedding_size 
			max_length : word_max_length 
		"""
		super(char_conv_embedding,self).__init__()
		self.conv_options = conv_options
		self.char_embedd = char_embedd
		self.max_length = max_length
		self.word_embedd_size = char_embedd*max_length
		self.in_channel = in_channel
		self.conv_module = nn.ModuleList([])

		for kernel_size,filter_num in conv_options.items():
			self.conv_module.append(nn.Conv2d(in_channels = in_channel,kernel_size=(kernel_size,char_embedd),out_channels=filter_num))
	
	def load_char_conv_embedding():
		raise NotImplementedError 
	def forward(self,x):
		"""
		Params:
			x: (Type:Torch.Tensor,Size : Batch x Time_step x max_length(?) x char_embedidng) ;

		Returns:
			output : Batch x Time_step x (total_channel)

		"""
		batch_size, time_step, max_length,char_embedding_size = x.size()
		
		x = x.view(batch_size*time_step,max_length,char_embedding_size).unsqueeze(1)
		output = []
		for conv_layer in self.conv_module:
			out = torch.max(conv_layer(x).squeeze(-1),-1)[0].squeeze(1) # Bacth*Time_step x num_channel
			output.append(out)
		output = torch.cat(output,dim=-1) # Batch*Time_step x (num_channel_1 + num_channel2 ...)
		return output 


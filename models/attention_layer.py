
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from .sub_layers import mlp_layers 
import numpy as np


class Attention_layer(nn.Module):
	def __init__(self,head_size,num_head,num_blocks,attention_mlp_layers,dropout_p= 0.1):
		super(Attention_layer,self).__init__()
		self.head_size = head_size
		self.num_head = num_head
		self.num_blocks = num_blocks
		self.attention_mlp_layers = attention_mlp_layers
		self.dropout = nn.Dropout(p=dropout_p)
		for i in range(num_blocks):
			setattr(self,'attention_block_{}'.format(i),Attention_block(self.head_size*self.num_head,self.head_size*self.num_head,self.num_head,dropout_p=dropout_p))
			setattr(self,'mlp_{}'.format(i),mlp_layers(self.head_size*self.num_head,attention_mlp_layers,dropout_p))
			setattr(self,'layer_norm_0_{}'.format(i),nn.LayerNorm(self.head_size*self.num_head))
			setattr(self,'layer_norm_1_{}'.format(i),nn.LayerNorm(self.head_size*self.num_head))
			
	def forward(self,inputs):
		"""
		args:
			inputs: B * memslots+1 * mem_size 
		returns:
			outputs: B* mem_slots+1 * mem_size 
			
		"""
		attention_maps = []
		
		for i in range(self.num_blocks):
			layer = getattr(self,'attention_block_{}'.format(i))
			new_memory,attention_map = layer(inputs,inputs,inputs)
			attention_maps.append(attention_map)
			
			layer_norm = getattr(self,'layer_norm_0_{}'.format(i))
			new_memory = layer_norm(inputs + new_memory)
			
			new_memory=self.dropout(new_memory)
			
			mlplayer = getattr(self,'mlp_{}'.format(i))
			layer_norm = getattr(self,'layer_norm_1_{}'.format(i))
			
			inputs  = layer_norm(new_memory+ mlplayer(new_memory))
		outputs = inputs
		return (outputs,attention_maps)



class Attention_block(nn.Module):
	"""
	"""
	def __init__(self,key_dim,value_dim,h,dropout_p = 0.1):

		super(Attention_block,self).__init__()
		self.key_dim = key_dim
		self.value_dim = value_dim
		self.n_heads = h 
		self.dropout_p = dropout_p
		self.Wq = nn.Parameter(torch.FloatTensor(self.n_heads,self.key_dim,int(self.key_dim/self.n_heads)))
		self.Wk = nn.Parameter(torch.FloatTensor(self.n_heads,self.key_dim,int(self.key_dim/self.n_heads)))
		self.Wv = nn.Parameter(torch.FloatTensor(self.n_heads,self.value_dim,int(self.value_dim/self.n_heads)))
		self.sc_dot_product_attentions = scaled_dot_product_attention(self.value_dim/self.n_heads,self.value_dim/self.n_heads,self.dropout_p)
		self.init_parameters()
		self.q_layer_norm  	= nn.LayerNorm(int(self.value_dim/self.n_heads))
		self.k_layer_norm	= nn.LayerNorm(int(self.value_dim/self.n_heads))
		self.v_layer_norm	= nn.LayerNorm(int(self.value_dim/self.n_heads)) 
	def init_parameters(self):

		nn.init.orthogonal_(self.Wq)
		nn.init.orthogonal_(self.Wk)
		nn.init.orthogonal_(self.Wv)
		
	def forward(self,q,k,v, mask =None):
		"""
		Attention block 
		composed of scaled dot Product
			Args :
				q : Batch*QL*k_dim
				k : B*KL*k_dim
				v : B*VL*v_dim
				mask :
			Returns:
				out : B 
				attention: 
		"""
		batch_size,q_length,q_dim =  q.size()
		_,k_length,k_dim =  k.size()
		_,v_length,v_dim =  v.size()
		# split 
		
		q_pj = q.repeat(self.n_heads,1,1).view(self.n_heads,-1,self.key_dim)   # H X B*L * K 
		k_pj = k.repeat(self.n_heads,1,1).view(self.n_heads,-1,self.key_dim)
		v_pj = v.repeat(self.n_heads,1,1).view(self.n_heads,-1,self.value_dim)
		
		# projection

		q_pj = torch.bmm(q_pj,self.Wq).view(-1,q_length,int(self.key_dim/self.n_heads)) # H*B X L X d-m
		k_pj = torch.bmm(k_pj,self.Wk).view(-1,k_length,int(self.key_dim/self.n_heads))
		v_pj = torch.bmm(v_pj,self.Wv).view(-1,v_length,int(self.value_dim/self.n_heads))

		# sclaed dot product
		outputs,attention =self.sc_dot_product_attentions(self.q_layer_norm(q_pj),self.k_layer_norm(k_pj),self.v_layer_norm(v_pj),mask) # H*B X L X d-v
		out = torch.cat(torch.split(outputs,batch_size,dim= 0),dim =-1) # B X L X d_model	
		
		
		return out,attention



class scaled_dot_product_attention(nn.Module):
	"""Scaled Dot-Prodcut attention layer class.
	"""
	def __init__(self,key_dim,value_dim,dropout_p = 0.1):
		super(scaled_dot_product_attention,self).__init__()
		#nn.softmax if upgrade pytorch version it need to replace dimension information
		self.key_dim = key_dim
		self.value_dim = value_dim
		self.dropout_p = dropout_p
		self.attention_dropout = nn.Dropout(p = self.dropout_p)

	def forward(self,q,k,v,mask):
		"""
		Scaled Dot Product attention layer 
		
			Args :
				q : (Batch*n_head) x q_lengths x q_dim
				k : (Batch*n_head) x k_lengths x k_dim
				v : (Batch*n_head) x v_lengths x v_dim
				mask : (Batch*n_head) x q_lengths x v_lengths 
			Returns :
				out : (Batch * n_head) x q_lengths x v_dims 
				attention : (Batch * n_heads) x q_lengths x k_lengths
		"""
		
		attention = torch.bmm(q,k.transpose(1,2))/np.sqrt(self.value_dim)  # (BxH) X Q_L X K_L
		if mask is not None:
			# fill -inf vlaue no attention 
			attention.data.masked_fill_(mask,-1e9)

		#attention = self.attention_dropout(F.softmax(attention,dim=2))
		out = self.attention_dropout(torch.bmm(attention,v)) #Calculate bmm B X Q_L X K_L and ( K_L X v_dim ) -> B X Q_L *V_dim 

		return out,attention

import torch
import torch.nn as nn
import torch.nn.functional as F 
from .attention_layer import Attention_layer 
import pdb
class RelationalRNN(nn.Module):
	def __init__(self,input_size,mem_slots,head_size,num_heads=1,num_blocks = 1,
		forget_bias = 1.0,input_bias =0.0,gate_style = 'unit',
		attention_mlp_layers = 2,dropout_p=0.2,key_size =None):
		super(RelationalRNN,self).__init__()
		self.mem_slots = mem_slots
		self.num_heads = num_heads
		self.head_size = head_size
		self.input_size = input_size
		self.mem_size  = head_size * num_heads
		self.num_units = self.mem_size * self.mem_slots
		#self.input_bias = input_bias
		#self.forget_bias = forget_bias
		
		if num_blocks < 1:
			raise ValueError('num_blocks must be >=1. But got: {}'.format(num_blocks))
		self._num_blocks = num_blocks

		if gate_style not in ['unit','memory',None]:
			raise ValueError(
				'gate_style must be one of [\'unit\',\'memory\',None].But got:{}'.format(gate_style)
				)
		self.gate_style = gate_style

		if attention_mlp_layers < 1:
			raise ValueError('attnetion mlp layers must be >1. But got:{}'.format(attention_mlp_layers))
		self.mhdpa_layers = Attention_layer(self.head_size,self.num_heads,num_blocks,attention_mlp_layers,dropout_p)

		# gate initialization
		self.input_linear  = nn.Linear(self.input_size,self.mem_size)
		self.forget_bias = nn.Parameter(torch.FloatTensor([forget_bias]))
		self.input_bias = nn.Parameter(torch.FloatTensor([input_bias]))

		if gate_style == 'unit':
			#self.gate_weight = nn.Paraemters(torch.randn(2,input_size+self.mem_size))
			#self.gate_bias = nn.Paraemters(torch.ones(2,1))

			self.gate_weight = nn.Parameter(torch.randn(2*self.mem_size,2))
			self.gate_bias = nn.Parameter(torch.ones(1,2))
		else:
			
			self.gate_weight = nn.Parameter(torch.randn(2*self.mem_size,2*self.mem_size))
			self.gate_bias = nn.Parameter(torch.ones(1,2*self.mem_size))


	def get_inital_memory(self,inputs):
		replace_eye = inputs.new_zeros(self.mem_slots,self.mem_size)
		nn.init.eye_(replace_eye)
		inital_state = replace_eye.unsqueeze(0).expand(inputs.size(0),-1,-1)
		return inital_state
	def forward(self,inputs,memory = None):
		"""
		parameters 
			inputs : Bactch * input_size 
			memory : Batch * mem_slots *mem_size  
		returns :
		"""

		if memory is None:
			memory = self.get_inital_memory(inputs)
		
		
		inputs = self.input_linear(inputs)
		
		if len(inputs.size()) ==2:
			inputs = inputs.unsqueeze(1)
		memory_plus_inputs  = torch.cat((memory,inputs),dim=1)

		next_memory,attention_maps = self.mhdpa_layers(memory_plus_inputs)
		n = inputs.size()[1] #get 

		next_memory = next_memory[:,:-n,:] # B *mem_slots* mem_size 

		#gating 
		gate_memory_inputs = torch.cat((memory,inputs.repeat(1,self.mem_slots,1)),dim=-1) # B*mem_slots * mem_size +input_size 
		gate_output = gate_memory_inputs.matmul(self.gate_weight) + self.gate_bias.unsqueeze(1) # b*mem_slots * gate_unit(2) or gate_mem(2* mem_size)
		input_gate_output, forget_gate_output = torch.chunk(gate_output,2,dim=-1) # each gate outputs size is B*memslot* gate_unit(1) or gate_mem(mem_size) = > B*mem_slots,mem_size/2 , B*mem_slots/mem_size/2
		input_gate_output = F.sigmoid(input_gate_output + self.input_bias)
		forget_gate_output = F.sigmoid(forget_gate_output+ self.forget_bias)

		next_memory =  input_gate_output*F.tanh(next_memory)
		next_memory += forget_gate_output*memory

		outputs = next_memory.view(inputs.size(0),-1)

		# out
		return outputs,next_memory,attention_maps 

class RelationalRNN_module(nn.Module):
	def __init__(self,*args,**kwargs):
		super(RelationalRNN_module,self).__init__()
		self.relationrnn = RelationalRNN(*args,**kwargs)
	def forward(self,inputs,memory =None):
		"""
		args:
			inputs : B * L * embedding_size
			memory : B * memslots * mem_size 
		returns:
			outputs : B* L * total_units 
			memory: B * memslots * mem_size
			total_attention_maps : L*B* (attention maps) --?

		"""
		
		batch_size,length,_ = inputs.size()
		total_attention_maps = []
		total_outputs = []
		for i in range(length):
			cur_input = inputs[:,i,:]
			output,memory,attention_maps = self.relationrnn(cur_input,memory)
			total_attention_maps.append(attention_maps)
			total_outputs.append(output.unsqueeze(0))
		outputs = torch.cat(total_outputs).permute(1,0,2)
		if batch_size != outputs.size(0):
			print ("[+] error raise")
		return outputs,memory,total_attention_maps


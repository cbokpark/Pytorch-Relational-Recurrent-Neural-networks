import torch
import torch.nn
from tensorboardX import SummaryWriter 
import torch.optim as optim
import torch.nn as nn 
import math 

import pdb

class Trainer:
	def __init__(self,model,embedding,epoch,train_data,loss,name,batch_size,vocab,
		validData  = None,testDataLoader =None,device=0 ,unroll_step = 100,clip_grad=0.25,
		save_freq =1000,optimizer_type='Adam',lr=1e-3):
		
		self.trainer_name = "Lanuge Model Trainer"
		self.name = name 
		
		self.model = model
		self.embedding = embedding

		self.train_data = train_data 
		
		self.validDataLoader = validData 
		self.testDataLoader = testDataLoader
		self.epoch = epoch
		self.save_freq = save_freq
		self.device = device
		self.criterion = loss
		self.unroll_step = unroll_step
		self.batch_size = batch_size
		
		self.vocab = vocab
		self.clip_grad = clip_grad

		self.summary_writer = SummaryWriter()
		self.valid = True if self.validDataLoader is not None else False 
		self.test = True if self.testDataLoader is not None else False
		self.projection_layer = nn.Sequential(nn.Linear(self.model.relationrnn.num_units,524),
								nn.Dropout(0.3),
								nn.Linear(524,vocab.size))

		self.total_iteration = 0 
		self.model.to(self.device)
		self.embedding.to(self.device)
		self.projection_layer.to(self.device)
		if optimizer_type == 'Adam':
			self.optimizer = getattr(optim, optimizer_type)(list(self.model.parameters()) +list(self.embedding.parameters())+list(self.projection_layer.parameters()) ,
				lr=lr,amsgrad =True)
		else:
			self.optimizer = getattr(optim, optimizer_type)(list(self.model.parameters()) +list(self.embedding.parameters())+list(self.projection_layer.parameters()) ,
				lr=lr,amsgrad =True)


		self.start_epoch = 0 
	def train(self):
		for epoch in range(self.start_epoch,self.epoch):
			result = self._train_epoch(epoch)


	def _train_epoch(self,epoch):
		self._train_mode()
		train_loss = 0 
		data_gen = self.train_data.iter_batches(self.batch_size,self.unroll_step)
		for batch_no,batch in enumerate(data_gen,start=1):
			#print ("total_iteration : {}".format(self.total_iteration))
			inputs = batch['tokens_characters'].to(self.device) # X Dict : token_ids , tokens_characters , next_token_id
			target = batch['next_token_id'].to(self.device)
			self.zero_grad()
			inputs = self.embedding(inputs)
			outputs,_,attention_maps=self.model(inputs)	
			output_logits = self.projection_layer(outputs) # B*L*vocab_size 
			output_logits = output_logits.view(-1,output_logits.size(-1))
			target = target.view(-1)
			loss = self.criterion(output_logits,target)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
			self.optimizer.step()
			train_loss += loss.item()*unroll_step
			if (self.total_iteration+1)%1000 == 0 :
				print ("[+] iteration :{} , avg_loss :{} , perpelxtiy: {:8.2f}".format(self.total_iteration,train_loss/1000,math.exp(train_loss/1000)))
				self._summary_writer()
				# add perpelxtiy value
				train_loss =0 
			if (self.total_iteration+1)%10000 == 0 :
				self.save_model()
			self.total_iteration +=1


	def _train_mode(self):
		self.model.train()
		self.embedding.train()
		self.projection_layer.train()
	def zero_grad(self):
		self.model.zero_grad()
		self.embedding.zero_grad()
		self.projection_layer.zero_grad()
	def _test(self,epoch):
		raise NotImplementedError

	def save_model(self,epoch):
		torch.save(self,self.model.satedict(),'./save_model/'+self.name+'_'+str(self.total_iteration)+'.path.tar')
		torch.save(self,self.embedding.satedict(),'./save_model/'+self.name+'_'+str(self.total_iteration)+'cnn.path.tar')
		torch.save(self,self.projection_layer.satedict(),'./save_model/'+self.name+'_'+str(self.total_iteration)+'projection.path.tar')
	def _summary_write(self,loss):
		self.summary_writer.add_scalar('data/loss',loss,self.iteration)
		for name,param in self.model.named_parameters():
			self.summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
			self.tensorboad_writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')

	def _eval_metric(self,output,target):
		raise NotImplementedError
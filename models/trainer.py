import torch
import torch.nn
from tensorboardX import SummaryWriter 

class Trainer:
	def __init__(self,model,embedding,epoch,trainDataLoade,loss,name,
		validDataLoader  = None,testDataLoader  =None,device=0 ,
		save_freq =1000,optimizer_type='Adam',lr=1e-3):
		self.name = name 
		self.model = relation_model
		self.embedding = embedding
		self.trainDataLoade = trainDataLoade
		self.validDataLoader = validDataLoader
		self.testDataLoader = testDataLoader
		self.epoch = epoch
		self.save_freq = save_freq
		self.device = device
		self.loss = loss

		self.summary_writer = SummaryWriter()
		self.valid = True if self.validDataLoader is not None else False 
		self.test = True if self.testDataLoader is not None else False
		if optimizer_type = 'Adam':
			self.optimizer = getattr(optim, optimizer_type)(self.model.parameters(),lr=lr,amsgrad =True)
		else:
			self.optimizer = getattr(optim, optimizer_type)(self.model.parameters(),lr=lr)
		self.total_iteration = 0 

		self.mode.to(self.device)
		self.start_epoch = 0 
	def train(self):
		for epoch in range(self.start_epoch,self.epoch):
			result = self._train_epoch(epoch)

	def _train_epoch(self,epoch):
		self.model.train()
		train_loss = 0 
		# load data
		# get loss value 
		# loss backward
		#optimzer 
		# print iteration
		# valid #iteration
		# summary_writer clal
	def _test(self,epoch):
		raise NotImplementedError

	def save_model(self,epoch)
		torch.save(self,self.model.satedict(),'./save_model/'+self.name+'_'+str(self.total_iteration)+'.path.tar')
	def _summary_write(self,loss):
		self.summary_writer.add_scalar('data/loss',loss,self.iteration)
		for name,param in self.model.named_parameters():
			self.summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
			self.tensorboad_writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')

	def _eval_metric(self,output,target):
		raise NotImplementedError
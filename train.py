import torch
import torch.nn as nn
import argparse 
from models.relation_rnn import RelationalRNN_module
from models.conv_module import CharEncodeNework
from models.trainer import Trainer 

from utils.data import load_vocab,BidirectionalLMDataset

import pdb 

def main(args):
    
    if args.gpu_num is not None :
        torch.cuda.set_device(args.gpu_num)
    if args.gpu_accelerate:
        torch.backends.cudnn.benchmark = True
    if args.gpu_num == -1:
        device = 'cpu'
    else:
        device = args.gpu_num

    vocab = load_vocab(args.vocab_file,50)
    
    prefix = args.train_prefix
    print ("[+] Load Language model data ")
    train_data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)
    print ("[+] Define Relation Recurrent Neural Networks")    
    model = RelationalRNN_module(input_size = 2048,mem_slots=1, head_size =int(2500/4),num_heads=4,gate_style='memory',attention_mlp_layers=5,dropout_p=0.5)
    conv_option ={1:32,2:32,3:64,4:128,5:256,6:512,7:1024}
    character_conv = CharEncodeNework(num_vocab=len(vocab._idx_to_char),max_length=50,
                        n_highway=2,char_embedd_size=5,conv_options=conv_option,use_bn =True)
    loss = nn.CrossEntropyLoss()
    print ("[+] Define Trainer")
    trainer = Trainer(model=model,
            embedding=character_conv,
            epoch=args.num_epoch,
            train_data = train_data
            ,loss =loss ,
            name=args.model_name,
            batch_size=args.batch_size,
            vocab=vocab, device =0,
            unroll_step=args.bptt,
            clip_grad=0.25 )
    
    trainer.train()
    print ("[+] Finished Training ")
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type = int, default = None)
    parser.add_argument('--num_epoch',type=int,default =50)
    parser.add_argument('--batch_size',type=int,default =128)
    parser.add_argument('--bptt',type=int,default = 100)
    parser.add_argument('--tensorboard_dirs',type=str,default ='./run')
    parser.add_argument('--gpu_accelerate',action='store_true')
    parser.add_argument('--save_model',type=str,default = './')
    parser.add_argument('--vocab_file',help='Vocabulary files')
    parser.add_argument('--train_prefix',help='Prefix for train false')
    parser.add_argument('--valid_prefix',default =None,help= 'prefix for valid ')
    parser.add_argument('--model_name',default ='experiment')
    parser_config = parser.parse_args()
    main(parser_config)
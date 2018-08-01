# Pytorch-Relational-Recurrent-Neural-networks
**[Adam Santoro, Ryan Faulkner, David Raposo, Jack Rae, Mike Chrzanowski, Theophane Weber, Daan Wierstra, Oriol Vinyals, Razvan Pascanu, Timothy Lillicrap, "Relational recurrent neural networks" arXiv preprint arXiv:1806.01822 (2018)](https://arxiv.org/abs/1806.01822).**


## Meta overview
This repository provides a PyTorch implementation of [Relational recurrent neural networks](https://arxiv.org/abs/1806.01822).


## Current update status
* [x] Supervised setting - language modeling 
* [] Supervised setting - Nth farthest problems 
* [x] Tensorboard loggings
* [] Langue modeling - memory efficient softmax 
* [x] Attention visualization (LSUN Church-outdoor)
* [x] Implemented core , self attention blocks , data loader 


## Results
TBD 
## Prerequisites 
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0](http://pytorch.org/)
* [tensorboardX](https://github.com/lanpa/tensorboardX)
* [Tesnrobard 1.2](https://github.com/tensorflow/tensorflow) : optional 


## Usage

#### 1. Clone the repository

```bash
$ git clone https://github.com/cheonbok94/Pytorch-Relational-Recurrent-Neural-networks.git
$ cd Pytorch-Relational-Recurrent-Neural-networks
$ pip install -r requirements.txt 
```

#### 2. Install datasets (CelebA or LSUN)
```bash
$ TBD
```

#### 3. Train 
##### (1) Train Language modeling 
```bash
$ python train.py --vocab_file ../data/vocab-2016-09-10.txt --train_prefix='../data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*' --gpu_num 0 --num_epoch 100 --gpu_accelerate --batch_size 6 --bptt 70

```

#### 4. Test
```bash
$ TBD....
```


#### 5.(optional) Tensorboard logging 


## Reference 















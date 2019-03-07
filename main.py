import argparse
import torch
import os
import sys
from solver import Solver

parser = argparse.ArgumentParser(description='Pytorch implementation of Classification Model')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--train', action='store_true', default=False, help='train model')
parser.add_argument('--test', action='store_true', default=False, help='test model')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='Define which dataset will be adopted (default: mnist) choices=> mnist, cifar10')
parser.add_argument('--train_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_epoch', type=int, default=50, help='num_epoch')
parser.add_argument('--lr', type=float, default=0.01, help='learning_rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint directory')
parser.add_argument('--dataset_dir', type=str, default='data', help='dataset directory')
args = parser.parse_args()

if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
if not os.path.isdir(args.checkpoint_dir+ '/' + args.dataset):
    os.mkdir(args.checkpoint_dir + '/' + args.dataset)
if not os.path.isdir(args.dataset_dir):
    os.mkdir(args.dataset_dir)

'''whether to use cuda'''
args.cuda = not args.no_cuda and torch.cuda.is_available()
'''print arguments'''
print(args)

def train():
    solver = Solver(args)
    for i in range(args.num_epoch):
        print("********************"+"Epoch: "+str(i)+"***************************")

        '''train network'''
        print("===========================Training===========================")
        losses = solver.train(i)

        '''print losses'''
        for _,key in enumerate(losses.keys()):
            print(str(key)+': '+str(torch.mean(torch.FloatTensor(losses[key]))))

        '''Save trained model'''
        print("===========================Save Model=============================")
        solver.save_model(path=args.dataset)

def test():
    solver = Solver(args)
    solver.load_model(path=args.dataset)
    test_result = solver.test()
    for _, key in enumerate(test_result.keys()):
        print(str(key) + ': ' + str(torch.mean(torch.FloatTensor(test_result[key]))))


if __name__=='__main__':
    if args.train:
        train()
    elif args.test:
        test()
    else:
        raise Exception("Specify wheter to train or test module")


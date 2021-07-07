import sys
import os

import warnings


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

def main():
    
    args = parser.parse_args()
    args.num_output=6400
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 100 #400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30

    for num_model in range(10):
      net=newModel().to(device)
      lr = 0.0001
      optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=5e-4)
      criterion = ContrastiveLoss()
      criterion_model = nn.CrossEntropyLoss(reduction='sum')
      best_acc=0
      EPOCH=250
      for epoch in range(EPOCH):
          loss_ls=[]
          loss1_ls=[]
          loss2_3_ls=[]
          t0=time.time()
          net.train()
          for seq1,seq2,label,label1,label2 in train_iter_cont:
                  output1=net(seq1)
                  output2=net(seq2)
                  output3=net.trainModel(seq1)
                  output4=net.trainModel(seq2)
                  loss1=criterion(output1, output2, label)
                  loss2=criterion_model(output3,label1)
                  loss3=criterion_model(output4,label2)
                  loss=loss1+loss2+loss3
      #             print(loss)
                  optimizer.zero_grad() 
                  loss.backward()
                  optimizer.step()
                  loss_ls.append(loss.item())
                  loss1_ls.append(loss1.item())
                  loss2_3_ls.append((loss2+loss3).item())


          net.eval() 
          with torch.no_grad(): 
              train_acc=evaluate_accuracy(train_iter,net)
              test_acc=evaluate_accuracy(test_iter,net)
          results=f"epoch: {epoch+1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
          results+=f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc,"red")}, time: {time.time()-t0:.2f}'
          print(results)
          to_log(results)
          if test_acc>best_acc:
              best_acc=test_acc
              torch.save({"best_acc":best_acc,"model":net.state_dict()},f'./Model/{num_model}.pl')
              print(f"best_acc: {best_acc}")
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    
        


    
if __name__ == '__main__':
    main()        





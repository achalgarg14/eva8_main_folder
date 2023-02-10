import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from tqdm import tqdm


def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc, lambda_l1=0):

  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss = 0
  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    y_pred = model(data)
    loss = F.cross_entropy(y_pred, target)

    if(lambda_l1 > 0):
      l1 = 0
      for p in model.parameters():
        l1 = l1 + p.abs().sum()
      loss = loss + lambda_l1*l1

    train_loss += loss.item()
    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  
  train_losses.append(train_loss/len(train_loader.dataset))
  train_acc.append(100*correct/len(train_loader.dataset))


def test(model, device, test_loader, test_losses, test_acc, epoch, target_acc=95, folder='EVA8/'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy_epoch = 100. * correct / len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    if(accuracy_epoch >= target_acc):
        model_name_file = "Model_" + str(epoch) + "_acc_" + str(round(accuracy_epoch,2)) + ".pth"
        path = "/content/drive/MyDrive/" + folder + model_name_file
        torch.save(model.state_dict(), path)
        print(f'Saved Model weights in file: {path}')
    return accuracy_epoch
  
def train_test_model(model, trainloader, testloader, EPOCHS=20, lr=0.001, optim = 'SGD', sched='StepLR', lambda_l1 = 0, target_acc=90, max_epoch=5, device='cpu', folder = 'EVA8/'):

  train_losses = []
  test_losses = []
  train_acc = []
  test_acc = []
  wrong_pred = []
  scheduler = None
  lambda_l1 = 0
  count =0
  model =  model.to(device)

  if(optim=='Adam'):
    optimizer = Adam(model.parameters(), lr=lr)
  elif(optim == 'SGD'):
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
  else:
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
  print(f'optimizer defined is {optim}')

  if sched == 'StepLR':
    scheduler = StepLR(optimizer, step_size=100, gamma=0.25)
    sch = True
  elif sched == 'OneCycleLR':
    scheduler = OneCycleLR(optimizer=optimizer, max_lr=0.05, epochs=EPOCHS, steps_per_epoch=len(trainloader), pct_start=max_epoch/EPOCHS, div_factor=10)
    sch = True
  else:
    sch = False

  for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, trainloader, optimizer, epoch, train_losses, train_acc, lambda_l1)
    
    if sch == True:
      scheduler.step()
    
    eval_test_acc = test(model, device, testloader, test_losses, test_acc, epoch, target_acc, folder)
    
    if(eval_test_acc >= target_acc):
      count +=1
      if count >2:
        break
  
  model_name_file = "Model_final_acc_" + str(round(eval_test_acc,2)) + ".pth"
  path = "/content/drive/MyDrive/" + folder + model_name_file
  torch.save(model.state_dict(), path)
  print(f'Saved Model weights in file: {path}')
  
  model.eval()
  
  for images, labels in testloader:
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    match = pred.eq(labels.view_as(pred)).to('cpu').numpy()
    for j, i in enumerate(match):
      if(i == False):
        wrong_pred.append((images[j], pred[j].item(), labels[j].item()))


  print(f'Total Number of incorrectly predicted images by model is {len(wrong_pred)}')
  return model, train_losses, test_losses, train_acc, test_acc, wrong_pred
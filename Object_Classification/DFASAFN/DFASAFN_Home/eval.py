import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

from utils import CLEFImage, print_args
from model.net import ResNet50_mod_name, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='data/OfficeHome/list')
parser.add_argument("--target", default='Art')
parser.add_argument("--batch_size", default=64)
parser.add_argument("--shuffle", default=False)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--snapshot", default="")
parser.add_argument("--epoch", default=300, type=int)
parser.add_argument("--result", default='record')
parser.add_argument("--class_num", default=65)
parser.add_argument("--task", default='None', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
args = parser.parse_args()
print_args(args)

result = open(os.path.join(args.result, "ImageCLEF_IAFN_" + args.task + '_' + args.post + '.' + args.repeat +"_score.txt"), "a")

target_root = 'data/OfficeHome/'+args.target
target_label = os.path.join(args.data_root, args.target+'.txt')
data_transform = transforms.Compose([
    transforms.Scale((256, 256)),
    transforms.CenterCrop((221, 221)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

t_set = CLEFImage(target_root, target_label, data_transform)
t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

netG = ResNet50_mod_name().cuda()
netF = ResClassifier(class_num=args.class_num, extract=False).cuda()
netG.eval()
netF.eval()
maxCorrect = 0
for epoch in range(1, args.epoch + 1):
    netG.load_state_dict(torch.load(os.path.join(args.snapshot, "ImageHome_IAFN_" + args.task + "_netG_" + args.post + '.' + args.repeat + '_'  + str(epoch) + ".pth")))
    netF.load_state_dict(torch.load(os.path.join(args.snapshot, "ImageHome_IAFN_" + args.task + "_netF_" + args.post + '.' + args.repeat + '_'  + str(epoch) + ".pth")))
    correct = 0
    tick = 0
    for (imgs, labels) in t_loader:
        tick += 1
        imgs = Variable(imgs.cuda())
        pred = netF(netG(imgs))
        pred = F.softmax(pred)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()
    correct = correct * 1.0 / len(t_set)
    if correct > maxCorrect:
        maxCorrect = correct
    print ("Epoch {0}: {1}".format(epoch, correct))
    result.write("Epoch " + str(epoch) + ": " + str(correct) + "\n")
print ("Max: {0}".format(maxCorrect))
result.write("Max: {0}".format(maxCorrect))
result.close()

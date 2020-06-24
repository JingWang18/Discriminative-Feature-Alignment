# run: python train_ent.py --task CP0 --source Clipart --target Product
import os
import argparse
import tqdm
import os
import argparse
import numpy as np
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from utils import CLEFImage, weights_init, print_args
from model.net import ResNet50_mod_name, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='data/OfficeHome/list')
parser.add_argument("--source", default='Clipart')
parser.add_argument("--target", default='RealWorld')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--snapshot", default="")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--class_num", default=65)
parser.add_argument("--extract", default=True)
parser.add_argument("--weight_L2norm", default=0.05)
parser.add_argument("--weight_entropy", default=0.1)
parser.add_argument("--dropout_p", default=0.5)
parser.add_argument("--task", default='None', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
parser.add_argument("--alpha", default=0.1, type=float)
parser.add_argument("--beta", default=10.0, type=float)
args = parser.parse_args()
print_args(args)

source_root = 'data/OfficeHome/'+args.source
source_label = os.path.join(args.data_root, args.source+'.txt')
target_root = 'data/OfficeHome/'+args.target
target_label = os.path.join(args.data_root, args.target+'.txt')

train_transform = transforms.Compose([
    transforms.Scale((256, 256)),
    transforms.RandomCrop((221, 221)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

source_set = CLEFImage(source_root, source_label, train_transform)
target_set = CLEFImage(target_root, target_label, train_transform)

source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

netG = ResNet50_mod_name().cuda()
state_dict = torch.load('model/resnet_model.pth')
netG.load_state_dict(state_dict)
netF = ResClassifier(class_num=args.class_num, extract=args.extract, dropout_p=args.dropout_p).cuda()
netF.apply(weights_init)


def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return args.weight_entropy * (entropy / float(p_softmax.size(0)))   


# initialize tensors
feat_t_recon = torch.ones([32, 3, 221, 221]).cuda()
feat_zn_recon = torch.ones([32, 3, 221, 221]).cuda()

# initialize a L1 loss for DAL
CriterionDAL = nn.L1Loss().cuda()

Tensor = torch.cuda.FloatTensor

opt_g = optim.SGD(netG.parameters(), lr=args.lr, weight_decay=0.0005)
opt_f = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
for epoch in range(1, args.epoch+1):
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)
    print(">>training " + args.task + " epoch : " + str(epoch))


    for i, (t_imgs, _) in tqdm.tqdm(enumerate(target_loader_iter)):
        try:
            s_imgs, s_labels = source_loader_iter.next()
        except:
            source_loader_iter = iter(source_loader)
            s_imgs, s_labels = source_loader_iter.next()

        if s_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
            continue

        s_imgs = Variable(s_imgs.cuda())
        s_labels = Variable(s_labels.cuda())     
        t_imgs = Variable(t_imgs.cuda())

        zn = Variable(Tensor(np.random.normal(0,1, (32, 2048))))
        
        opt_g.zero_grad()
        opt_f.zero_grad()

        s_bottleneck = netG(s_imgs)
        t_bottleneck = netG(t_imgs)      
        
        s_fc2_emb, s_logit = netF(s_bottleneck)
        t_fc2_emb, t_logit = netF(t_bottleneck)
        
        s_cls_loss = get_cls_loss(s_logit, s_labels)

        #kl-divergence
        feat_s_kl = s_bottleneck.view(-1,2048)
        loss_kld_s = F.kl_div(F.log_softmax(feat_s_kl), F.softmax(zn))

        #distribution alignment loss (DAL)
        loss_dal= CriterionDAL(feat_t_recon, feat_zn_recon)
        
        t_prob = F.softmax(t_logit)
        t_entropy_loss = get_entropy_loss(t_prob)
        
        #updated loss function
        loss = s_cls_loss + t_entropy_loss + args.alpha * loss_kld_s  + args.beta * loss_dal
        loss.backward()
        
        if (i+1) % 5 == 0:
            print ("cls_loss: %.4f, entropy_loss: %.4f" % (s_cls_loss.item(), t_entropy_loss.item()))
        
        opt_g.step()
        opt_f.step()

        # calculate decoded samples for the next iteration
        feat_t_recon = netG(t_imgs, is_deconv=True).detach()
        feat_zn_recon = netG.decode(zn).detach()

    
    if epoch % 1 == 0:
        torch.save(netG.state_dict(), os.path.join(args.snapshot, "ImageHome_IAFN_" + args.task + "_netG_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth"))
        torch.save(netF.state_dict(), os.path.join(args.snapshot, "ImageHome_IAFN_" + args.task + "_netF_" + args.post + '.' + args.repeat + '_'  + str(epoch) + ".pth"))


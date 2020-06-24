from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import Generator, Classifier
from datasets.dataset_read import dataset_read

import numpy as np


# Training settings
class Solver(object):
    def __init__(self, args, batch_size=256, source='mnist',
                 target='usps', learning_rate=0.02, interval=100, optimizer='momentum'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.alpha = args.alpha
        self.beta = args.beta
        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False
        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(source, target, self.batch_size, scale=self.scale,
                                                        all_use=self.all_use)
        print('load finished!')
        self.G = Generator(source=source, target=target)
        self.C = Classifier(source=source, target=target)

        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.02, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_c = optim.SGD(self.C.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c = optim.Adam(self.C.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()

    def get_entropy_loss(self, p_softmax):
        mask = p_softmax.ge(0.000001)
        mask_out = torch.masked_select(p_softmax, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return 0.1 * (entropy / float(p_softmax.size(0))) 

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        # initialze a L1 loss for DAL
        criterionDAL = nn.L1Loss().cuda()
        
        self.G.train()
        self.C.train()
        torch.cuda.manual_seed(1)

        Tensor = torch.cuda.FloatTensor

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = Variable(label_s.long().cuda())

            # for mnist or usps (source) 
            zn = Variable(Tensor(np.random.normal(0,1, (4096, 48))))
            # for svhn (source)
            #zn = Variable(Tensor(np.random.normal(0,1, (16384, 128))))

            img_s = Variable(img_s)
            img_t = Variable(img_t)

            self.reset_grad()

            feat_s = self.G(img_s)
            output_s = self.C(feat_s)
            feat_t = self.G(img_t)
            output_t = self.C(feat_t)

            # for mnist or usps (source)
            feat_s_kl = feat_s.view(-1,48)
            # for svhn (source)
            #feat_s_kl = feat_s.view(-1,128)

            loss_kld_s = F.kl_div(F.log_softmax(feat_s_kl), F.softmax(zn))

            loss_s = criterion(output_s, label_s)

            loss = loss_s + self.alpha * loss_kld_s
            loss.backward()

            self.opt_g.step()
            self.opt_c.step()
            self.reset_grad()

            feat_t = self.G(img_t)
            output_t = self.C(feat_t)
            feat_t_recon = self.G(img_t, is_deconv= True)

            feat_zn_recon = self.G.decode(zn)
            # DAL 
            loss_dal = criterionDAL(feat_t_recon, feat_zn_recon) 

            # entropy loss
            t_prob = F.softmax(output_t)
            t_entropy_loss = self.get_entropy_loss(t_prob)

            loss = t_entropy_loss + self.beta * loss_dal
            loss.backward()

            self.opt_g.step()
            self.opt_c.step()
            self.reset_grad()

            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t   Entropy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s.item(), t_entropy_loss.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s\n' % (t_entropy_loss.item(), loss_s.item()))
                    record.close()
            torch.save(self.G,
                        '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C.eval()

        test_loss = 0
        correct = 0
        size = 0
        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat = self.G(img)
            output = self.C(feat)

            test_loss += F.nll_loss(output, label).item()
            pred = output.data.max(1)[1]

            k = label.data.size()[0]
            correct += pred.eq(label.data).cpu().sum()

            size += k
        test_loss = test_loss / size
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C: {}/{} ({:.0f}%) \n'.format(
                test_loss, correct, size,
                100. * correct / size))
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C,
                       '%s/%s_to_%s_model_epoch%s_C.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s\n' % (float(correct) / size))
            record.close()

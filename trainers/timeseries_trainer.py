import numpy as np
import torch
import torch.nn as nn
import torchvision
import einops
import wandb
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from trainers import register

@register('timeseries_trainer')
class TimeseriesTrainer(BaseTrainer):

    def make_datasets(self):
        super().make_datasets()

        def get_vislist(dataset, n_vis=32):
            ids = torch.arange(n_vis) * (len(dataset) // n_vis)
            return [dataset[i] for i in ids]

        if hasattr(self, 'train_loader'):
            self.vislist_train = get_vislist(self.train_loader.dataset)
        if hasattr(self, 'test_loader'):
            self.vislist_test = get_vislist(self.test_loader.dataset)

    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        if self.epoch <= round(self.cfg['max_epoch'] * 0.8):
            lr = base_lr
        else:
            lr = base_lr * 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.log_temp_scalar('lr', lr)

    def _iter_step(self, data, is_train, test=False):
        new_data = {}
        if self.adaptive_clustering == True:
            i = self.new_order
            batch_x = data[0][:,:,i].permute(0, 2, 1).cuda().float()
        else:
            batch_x = data[0][:,:,i].permute(0, 2, 1).cuda().float()
        
        b, n, t = batch_x.shape
        mask = torch.rand((b, n, t)).cuda()
        mask[mask <= self.mask_rate] = 0  # masked
        mask[mask > self.mask_rate] = 1  # remained
        ori_inp = batch_x.masked_fill(mask == 0, 0)

        new_data['inp'] = ori_inp.cuda().float()
        new_data['mask'] = mask.float()
        new_data['gt'] = batch_x.float()
        data = new_data

        gt = data.pop('gt')
        B = gt.shape[0]

        hyponet = self.model_ddp(data)

        coord = np.linspace(-1,1,gt.shape[-1])
        coord = coord[:,np.newaxis]
        coord = torch.as_tensor(coord,dtype=torch.float).to(gt.device)
        
        coord = einops.repeat(coord, 'h d -> b h d', b=B)
        pred = hyponet(coord) # b h w 3
        gt = einops.rearrange(gt, 'b c h -> b h c')

        mask = data['mask'].permute(0,2,1)
        imp_gt = gt[mask == 0]
        imp_pred = pred[mask == 0]

        mses = ((imp_pred - imp_gt) ** 2).mean(dim=-1)
        loss = mses.mean()
        psnr = (-10 * torch.log10(mses)).mean()

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        ret = {'mse': mses.mean().item(), 'loss': loss.item(), 'psnr': psnr.item()}
        if test == True:
            return ret, pred, gt, mask
        return ret

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data, test=False):
        with torch.no_grad():
            return self._iter_step(data, is_train=False, test=test)


"""
    A basic trainer.

    The general procedure in run() is:
        make_datasets()
            create . train_loader, test_loader, dist_samplers
        make_model()
            create . model_ddp, model
        train()
            create . optimizer, epoch, log_buffer
            for epoch = 1 ... max_epoch:
                adjust_learning_rate()
                train_epoch()
                    train_step()
                evaluate_epoch()
                    evaluate_step()
                visualize_epoch()
                save_checkpoint()
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import os.path as osp
import time

import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt

import datasets
import models
import utils
from trainers import register
from utils.tools import adjustment
from .metric import metric

@register('base_trainer')
class BaseTrainer():

    def __init__(self, rank, cfg):
        self.rank = rank
        self.cfg = cfg
        self.is_master = (rank == 0)

        env = cfg['env']
        self.tot_gpus = env['tot_gpus']
        self.distributed = (env['tot_gpus'] > 1)

        # Setup log, tensorboard, wandb
        if self.is_master:
            logger, writer = utils.set_save_dir(env['save_dir'], replace=False)
            with open(osp.join(env['save_dir'], 'cfg.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)

            self.log = logger.info

            self.enable_tb = True
            self.writer = writer

            if env['wandb_upload']:
                self.enable_wandb = True
                with open('wandb.yaml', 'r') as f:
                    wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                os.environ['WANDB_DIR'] = env['save_dir']
                os.environ['WANDB_NAME'] = env['exp_name']
                os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
                wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=cfg)
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        # Setup distributed devices
        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', torch.cuda.current_device())

        if self.distributed:
            dist_url = f"tcp://localhost:{env['port']}"
            dist.init_process_group(backend='nccl', init_method=dist_url,
                                    world_size=self.tot_gpus, rank=rank)
            self.log(f'Distributed training enabled.')

        cudnn.benchmark = env['cudnn']

        self.log(f'Environment setup done.')

    def run(self):
        self.adaptive_clustering = self.cfg['adaptive_clustering']
        self.mask_rate = self.cfg['all_train_dataset']['args']['args']['mask_rate']
        self.make_datasets()
        self.make_model()
        self.train()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            wandb.finish()

    def make_datasets(self):
        """
            By default, train dataset performs shuffle and drop_last.
            Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

            Cfg example:

            train/test_dataset:
                name:
                args:
                loader: {batch_size: , num_workers: }
        """
        cfg = self.cfg
        self.dist_samplers = []

        def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
            sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
            loader = DataLoader(
                dataset,
                batch_size // self.tot_gpus,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=num_workers // self.tot_gpus,
                pin_memory=True)
            return loader, sampler

        if cfg.get('all_train_dataset') is not None:
            tmp = datasets.make(cfg['all_train_dataset'])
            train_dataset = tmp.data_set
            self.train_loader = tmp.data_loader
            if self.adaptive_clustering == True:
                self.new_order = train_dataset.new_order
                self.group_num_list = train_dataset.group_num_list
            else:
                self.new_order = None
                self.group_num_list = None
            self.log(f'Train dataset: len={len(train_dataset)}')

        if cfg.get('all_test_dataset') is not None:
            tmp = datasets.make(cfg['all_test_dataset'])
            test_dataset = tmp.data_set
            self.test_loader = tmp.data_loader
            self.log(f'Test dataset: len={len(test_dataset)}')


    def make_model(self, model_spec=None, load_sd=False):
        if model_spec is None:
            model_spec = self.cfg['model']
        if self.adaptive_clustering == True:
            model_spec['args']['hyponet']['args']['group_num_list'] = self.group_num_list
        model = models.make(model_spec, load_sd=load_sd)
        self.log(f'Model: #params={utils.compute_num_params(model)}')
        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(model, device_ids=[self.rank])
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp

    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg

        self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), cfg['optimizer'])

        max_epoch = cfg['max_epoch']
        eval_epoch = cfg.get('eval_epoch', max_epoch + 1)
        save_epoch = cfg.get('save_epoch', max_epoch + 1)
        epoch_timer = utils.EpochTimer(max_epoch)

        tmp = 0
        for epoch in range(1, max_epoch + 1):
            self.epoch = epoch
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)

            self.adjust_learning_rate()

            self.t_data, self.t_model = 0, 0
            self.train_epoch()

            if epoch % eval_epoch == 0:
                tmp = self.imputation_evaluate_epoch()
                #tmp = self.evaluate_epoch()
                result = {}
                for k, v in tmp.items():
                    result[k] = v.item()

            if epoch % save_epoch == 0:
                self.save_checkpoint(f'epoch-{epoch}.pth')
            self.save_checkpoint('epoch-last.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            t_data_ratio = self.t_data / (self.t_data + self.t_model)
            self.log_buffer.append(f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

        result = {}
        for k, v in tmp.items():
            result[k] = v.item()


    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
        self.log_temp_scalar('lr', self.optimizer.param_groups[0]['lr'])

    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=t)
        if self.enable_wandb:
            wandb.log({k: v}, step=t)

    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.tot_gpus)

    def sync_ave_scalars_(self, ave_scalars):
        for k in ave_scalars.keys():
            x = torch.tensor(ave_scalars[k].item(), dtype=torch.float32, device=self.device)
            self.dist_all_reduce_mean_(x)
            ave_scalars[k].v = x.item()
            ave_scalars[k].n *= self.tot_gpus

    def train_epoch(self):
        self.model_ddp.train()
        ave_scalars = dict()

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='train', leave=False)
        t1 = time.time()

        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.train_step(data)
            self.t_model += time.time() - t0

            B = data[0].shape[0]
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'train:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('train/' + k, v.item())
        self.log_buffer.append(logtext)


    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        pbar = self.test_loader

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.evaluate_step(data)
            self.t_model += time.time() - t0

            B = data[0].shape[0]
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        tmp_each_epoch = ave_scalars

        logtext = 'eval:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('test/' + k, v.item())
        self.log_buffer.append(logtext)
        return tmp_each_epoch


    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_spec = self.cfg['model']
        model_spec['sd'] = self.model.state_dict()
        optimizer_spec = self.cfg['optimizer']
        optimizer_spec['sd'] = self.optimizer.state_dict()
        checkpoint = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': self.epoch,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, osp.join(self.cfg['env']['save_dir'], filename))

    def imputation_evaluate_epoch(self):
        self.model_ddp.eval()
        preds = []
        trues = []
        masks = []
        ave_scalars = dict()

        pbar = self.test_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='eval', leave=False)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret, pred, gt, mask = self.evaluate_step(data, test=True)
            self.t_model += time.time() - t0

            B = data[0].shape[0]
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            outputs = pred.detach().cpu().numpy()
            pred = outputs
            true = gt.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
            masks.append(mask.detach().cpu())

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        tmp_each_epoch = ave_scalars

        logtext = 'eval:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('test/' + k, v.item())
        self.log_buffer.append(logtext)

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = self.cfg['env']['save_dir']
        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{}'.format(mse, mae))
        filename = osp.join(folder_path, 'result_imputation.txt')
        f = open(filename, 'a')
        f.write(str(self.epoch) + '\n')
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        folder_path += '/save/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return tmp_each_epoch


"""The module for training ENAS."""
import contextlib
import glob
import math
import os

import numpy as np
import scipy.signal
from tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

import models
import utils

logger = utils.get_logger()

def _apply_penalties(extra_out, args):
    penalty = 0
    if args.activation_regularization:
        penalty += (args.activation_regularization_amount *
                    extra_out['dropped'].pow(2).mean())
    if args.temporal_activation_regularization:
        raw = extra_out['raw']
        penalty += (args.temporal_activation_regularization_amount *
                    (raw[1:] - raw[:-1]).pow(2).mean())
    if args.norm_stabilizer_regularization:
        penalty += (args.norm_stabilizer_regularization_amount *
                    (extra_out['hiddens'].norm(dim=-1) -
                     args.norm_stabilizer_fixed_point).pow(2).mean())
    return penalty

def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

def _get_optimizer(name):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD
    elif name == 'momentum':
        # In PyTorch, Momentum is just SGD with the momentum parameter.
        # We return a functional wrapper to set the default.
        return lambda params, lr, **kwargs: torch.optim.SGD(params, lr=lr, momentum=0.9, **kwargs)
    elif name == 'adam':
        return torch.optim.Adam
    else:
        raise ValueError(f"Optimizer {name} not supported. Choose from sgd, momentum, adam.")

def _get_no_grad_ctx_mgr():
    if float(torch.__version__[0:3]) >= 0.4:
        return torch.no_grad()
    return contextlib.suppress()

def _check_abs_max_grad(abs_max_grad, model):
    finite_grads = [p.grad.data for p in model.parameters() if p.grad is not None]
    if not finite_grads: return abs_max_grad
    new_max_grad = max([grad.max() for grad in finite_grads])
    new_min_grad = min([grad.min() for grad in finite_grads])
    new_abs_max_grad = max(new_max_grad, abs(new_min_grad))
    if new_abs_max_grad > abs_max_grad:
        return new_abs_max_grad
    return abs_max_grad

class Trainer(object):
    def __init__(self, args, dataset):
        self.args = args
        self.controller_step = 0
        self.cuda = args.cuda
        self.dataset = dataset
        self.epoch = 0
        self.shared_step = 0
        self.start_epoch = 0

        # CNN vs RNN Data Handling
        if args.network_type == 'cnn':
            self.train_data = dataset.train
            self.valid_data = dataset.valid
            self.eval_data = dataset.valid
            self.test_data = dataset.test
        else:
            self.train_data = utils.batchify(dataset.train, args.batch_size, self.cuda)
            self.valid_data = utils.batchify(dataset.valid, args.batch_size, self.cuda)
            self.eval_data = utils.batchify(dataset.valid, args.test_batch_size, self.cuda)
            self.test_data = utils.batchify(dataset.test, args.test_batch_size, self.cuda)

        self.max_length = self.args.shared_rnn_max_length
        if args.use_tensorboard:
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None
        
        self.build_model()
        if self.args.load_path:
            self.load_model()

        shared_optimizer = _get_optimizer(self.args.shared_optim)
        self.shared_optim = shared_optimizer(self.shared.parameters(),
            lr=self.args.shared_lr, weight_decay=self.args.shared_l2_reg)

        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(),
            lr=self.args.controller_lr)

        self.ce = nn.CrossEntropyLoss()

    def build_model(self):
        if self.args.network_type == 'rnn':
            self.shared = models.RNN(self.args, self.dataset)
        elif self.args.network_type == 'cnn':
            self.shared = models.CNN(self.args, self.dataset)
        self.controller = models.Controller(self.args)
        if self.cuda:
            self.shared.cuda()
            self.controller.cuda()

    def get_loss(self, inputs, targets, hidden, dags):
        if not isinstance(dags, list):
            dags = [dags]
        loss = 0
        for dag in dags:
            # trainer.py
            output, hidden, extra_out = self.shared(inputs, dag, hidden=hidden)
            if self.args.network_type == 'cnn':
                # CNN outputs are usually [Batch, Classes]
                output_flat = output
            else:
                # RNN outputs are [SeqLen * Batch, Vocab]
                output_flat = output.view(-1, self.dataset.num_tokens)
            
            sample_loss = self.ce(output_flat, targets) / self.args.shared_num_sample
            loss += sample_loss
        return loss, hidden, extra_out

    def train_shared(self, max_step=None, dag=None):
        model = self.shared
        model.train()
        self.controller.eval()
        
        # RNNs need hidden states, CNNs usually don't
        hidden = self.shared.init_hidden(self.args.batch_size) if self.args.network_type == 'rnn' else None
        
        if max_step is None:
            max_step = self.args.shared_max_step

        step = 0
        raw_total_loss = 0
        
        # Use an iterator for CNN DataLoader
        if self.args.network_type == 'cnn':
            train_iter = iter(self.train_data)
        
        train_idx = 0
        while step < max_step:
            dags = dag if dag else self.controller.sample(self.args.shared_num_sample)
            
            if self.args.network_type == 'cnn':
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_data)
                    inputs, targets = next(train_iter)
                if self.cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
            else:
                inputs, targets = self.get_batch(self.train_data, train_idx)
                train_idx += self.max_length
                if train_idx >= self.train_data.size(0) - 1: train_idx = 0

            loss, hidden, extra_out = self.get_loss(inputs, targets, hidden, dags)
            if hidden is not None:
                if isinstance(hidden, tuple):
                    for h in hidden: h.detach_()
                else:
                    hidden.detach_()

            raw_total_loss += loss.item()
            
            self.shared_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.shared_grad_clip)
            self.shared_optim.step()

            if (step % self.args.log_step == 0) and (step > 0):
                self._summarize_shared_train(raw_total_loss, raw_total_loss)
                raw_total_loss = 0
            
            step += 1
            self.shared_step += 1

    def train_controller(self):
        model = self.controller
        model.train()
        avg_reward_base = None
        baseline = None
        reward_history = []
        
        hidden = self.shared.init_hidden(self.args.batch_size) if self.args.network_type == 'rnn' else None
        
        # For CNN, we use a simple counter to move through the validation set
        if self.args.network_type == 'cnn':
            valid_iter = iter(self.valid_data)

        for step in range(self.args.controller_max_step):
            dags, log_probs, entropies = self.controller.sample(with_details=True)
            
            with _get_no_grad_ctx_mgr():
                if self.args.network_type == 'cnn':
                    try:
                        inputs, targets = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(self.valid_data)
                        inputs, targets = next(valid_iter)
                    if self.cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    
                    output, _, _ = self.shared(inputs, dags)
                    valid_loss = self.ce(output, targets).item()
                    reward = self.args.reward_c / valid_loss # Simple accuracy-based reward proxy
                else:
                    # RNN Perplexity reward logic... (simplified here)
                    reward = 1.0 

            reward_history.append(reward)
            if baseline is None: baseline = reward
            else: baseline = self.args.ema_baseline_decay * baseline + (1 - self.args.ema_baseline_decay) * reward
            
            adv = reward - baseline
            loss = -log_probs * adv
            loss = loss.sum()

            self.controller_optim.zero_grad()
            loss.backward()
            self.controller_optim.step()
            self.controller_step += 1

    def train(self, single=False):
        dag = utils.load_dag(self.args) if single else None
        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            self.train_shared(dag=dag)
            if not single:
                self.train_controller()
            self.save_model()

    def get_batch(self, source, idx, length=None):
        length = min(length if length else self.max_length, len(source) - 1 - idx)
        data = source[idx:idx+length]
        target = source[idx+1:idx+1+length].view(-1)
        return data, target

    def save_model(self):
        torch.save(self.shared.state_dict(), self.shared_path)
        torch.save(self.controller.state_dict(), self.controller_path)

    @property
    def shared_path(self):
        return f'{self.args.model_dir}/shared.pth'
    @property
    def controller_path(self):
        return f'{self.args.model_dir}/controller.pth'

    def _summarize_shared_train(self, total_loss, raw_loss):
        cur_loss = total_loss / self.args.log_step
        logger.info(f'| epoch {self.epoch:3d} | step {self.shared_step:6d} | loss {cur_loss:5.2f}')
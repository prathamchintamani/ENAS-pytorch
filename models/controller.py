"""A module with NAS controller-related code."""
import collections
import os
import torch
import torch.nn.functional as F
import utils
from utils import Node

def _construct_dags(prev_nodes, activations, func_names, num_blocks):
    """
    Constructs a List-based DAG for the Shared CNN.
    
    The child model's _resolve_dag expects a List of Node objects.
    Because shared_cnn.py handles the linear chain logic internally,
    we only need to provide the operation names for each node id.
    """
    dags = []
    for nodes, func_ids in zip(prev_nodes, activations):
        dag_list = []
        
        for i in range(num_blocks):
            # Convert tensor index to python scalar
            op_id = utils.to_item(func_ids[i])
            op_name = func_names[op_id]
            
            # shared_cnn.py uses the 'name' attribute to look up the operation
            dag_list.append(Node(id=i, name=op_name))
            
        dags.append(dag_list)
    return dags


class Controller(torch.nn.Module):
    def __init__(self, args):
        super(Controller, self).__init__()
        self.args = args

        if self.args.network_type == 'rnn':
            self.num_tokens = [len(args.shared_rnn_activations)]
            for idx in range(self.args.num_blocks):
                self.num_tokens += [idx + 1, len(args.shared_rnn_activations)]
            self.func_names = args.shared_rnn_activations
        elif self.args.network_type == 'cnn':
            # For the current SharedCNN micro-search, we sample one op per node.
            # This creates a decoder for each of the nodes in the cell.
            self.num_tokens = [len(args.shared_cnn_types)] * self.args.num_blocks
            self.func_names = args.shared_cnn_types

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens, args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)

        self.decoders = torch.nn.ModuleList()
        for size in self.num_tokens:
            self.decoders.append(torch.nn.Linear(args.controller_hid, size))

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self, inputs, hidden, block_idx, is_embed):
        if not is_embed:
            # inputs is a LongTensor of indices [Batch]
            # encoder turns indices into vectors [Batch, HID]
            embed = self.encoder(inputs)
        else:
            # inputs is already a hidden-sized vector (initial zeros)
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature

        if self.args.mode == 'train':
            logits = (self.args.tanh_c * torch.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [Batch, HID]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = [] # Kept for interface parity, though not used for linear CNNs

        for block_idx in range(len(self.num_tokens)):
            # First step uses the zero-vector (is_embed=True)
            # Subsequent steps use the action index (is_embed=False)
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # Prepare the next input: shift index to unique embedding space for each block
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:block_idx]),
                requires_grad=False)

            activations.append(action[:, 0])

        # Stack sampled operations
        activations = torch.stack(activations).transpose(0, 1)
        
        # In this linear mode, prev_nodes are not sampled but inferred in _construct_dags
        fake_prev_nodes = [torch.zeros(batch_size)] * self.args.num_blocks

        dags = _construct_dags(fake_prev_nodes,
                               activations,
                               self.func_names,
                               self.args.num_blocks)

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag, os.path.join(save_dir, f'graph{idx}.png'))

        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))
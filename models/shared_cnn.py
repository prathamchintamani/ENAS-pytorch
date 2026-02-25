# models/shared_cnn.py
"""
Shared CNN (Child Model) for ENAS Micro Search Space on CIFAR-10.
Implements the weight-sharing child network whose architecture is
dynamically determined by the Controller's sampled DAG at each forward pass.

Reference: Pham et al., "Efficient Neural Architecture Search via Parameter Sharing"
           (https://arxiv.org/abs/1802.03268)

Compatibility fixes applied
───────────────────────────
  Fix 1 │ Node namedtuple mismatch
         │   utils.py defines  Node = namedtuple('Node', ['id', 'name'])
         │   _run_stage needs  .op_index  and  .inputs
         │   → _resolve_dag() converts any Controller-emitted Node into an
         │     internal _FullNode(id, op_index, inputs) by mapping the string
         │     'name' through OP_NAME_TO_IDX, and by inferring a sensible
         │     linear edge topology when .inputs is absent.
         │
  Fix 2 │ Missing extra_out
         │   trainer.py unpacks:
         │     output, hidden, extra_out = self.shared(inputs, dag, hidden=hidden)
         │   → forward() now returns (logits, None, {}) — the empty Dict is a
         │     neutral extra_out that the Trainer can safely inspect or ignore.
"""

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F   # noqa: F401  (available for custom use)
from typing import List, Dict, Callable, Tuple


# ============================================================================
# Primitive operation builders
# ============================================================================

def conv3x3(C_in: int, C_out: int, stride: int = 1) -> nn.Sequential:
    """Standard 3×3 convolution — 'same' padding (stride=1)."""
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=3, stride=stride,
                  padding=1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
    )


def conv5x5(C_in: int, C_out: int, stride: int = 1) -> nn.Sequential:
    """Standard 5×5 convolution — 'same' padding (stride=1)."""
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=5, stride=stride,
                  padding=2, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
    )


def sep_conv3x3(C_in: int, C_out: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable 3×3: depthwise → pointwise, both BN+ReLU."""
    return nn.Sequential(
        nn.Conv2d(C_in, C_in, kernel_size=3, stride=stride,
                  padding=1, groups=C_in, bias=False),
        nn.BatchNorm2d(C_in),
        nn.ReLU(inplace=True),
        nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
    )


def sep_conv5x5(C_in: int, C_out: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable 5×5: depthwise → pointwise, both BN+ReLU."""
    return nn.Sequential(
        nn.Conv2d(C_in, C_in, kernel_size=5, stride=stride,
                  padding=2, groups=C_in, bias=False),
        nn.BatchNorm2d(C_in),
        nn.ReLU(inplace=True),
        nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
    )


def max_pool3x3(C_in: int, C_out: int, stride: int = 1) -> nn.Sequential:
    """3×3 max-pool with 'same' padding; 1×1 proj when C_in != C_out."""
    layers: List[nn.Module] = [
        nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
    ]
    if C_in != C_out:
        layers += [
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


def avg_pool3x3(C_in: int, C_out: int, stride: int = 1) -> nn.Sequential:
    """3×3 average-pool with 'same' padding; 1×1 proj when C_in != C_out."""
    layers: List[nn.Module] = [
        nn.AvgPool2d(kernel_size=3, stride=stride,
                     padding=1, count_include_pad=False)
    ]
    if C_in != C_out:
        layers += [
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


# ============================================================================
# Operation registry
# ============================================================================

OP_NAMES: List[str] = [
    'conv3x3',
    'conv5x5',
    'sep_conv3x3',
    'sep_conv5x5',
    'max_pool3x3',
    'avg_pool3x3',
]

# ── FIX 1 (part a) ──────────────────────────────────────────────────────────
# The Controller stores Node.name (a string such as 'conv3x3').
# _run_stage works with integer indices for O(1) dispatch into nn.ModuleList.
# This Dict bridges the two representations without modifying utils.py.
OP_NAME_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(OP_NAMES)}

OP_BUILDERS: Dict[str, Callable] = {
    'conv3x3':     conv3x3,
    'conv5x5':     conv5x5,
    'sep_conv3x3': sep_conv3x3,
    'sep_conv5x5': sep_conv5x5,
    'max_pool3x3': max_pool3x3,
    'avg_pool3x3': avg_pool3x3,
}


# ============================================================================
# Internal full-node descriptor
# ============================================================================

# utils.py (unchanged):   Node = namedtuple('Node', ['id', 'name'])
# _FullNode is only used inside this file; it is never exported.
_FullNode = collections.namedtuple('_FullNode', ['id', 'op_index', 'inputs'])


def _resolve_dag(dag: List) -> List:
    """
    Convert a raw Controller DAG into a List of _FullNode objects.

    Handles three possible node formats emitted by different controllers:

      Format A  Node(id, name)          ← utils.py default (most common)
      Format B  Node(id, name, inputs)  ← extended controller with edges
      Format C  _FullNode(id, op_index, inputs) ← already resolved (pass-through)

    Edge inference when `.inputs` is absent
    ───────────────────────────────────────
    The micro search space in the ENAS paper builds a DAG where each new
    node can attend to *any* previous node.  When the Controller does not
    supply edge information (Format A), we default to a safe linear chain:
        node 0  ← stage input  (index –1)
        node k  ← node k-1

    This matches the simplest valid micro-cell and keeps the child model
    runnable even with a minimal controller implementation.

    Args:
        dag: List of Node or _FullNode objects from the Controller.

    Returns:
        List of _FullNode with fully populated id, op_index, inputs.

    Raises:
        ValueError  if a node name is not in OP_NAME_TO_IDX.
        AttributeError  if a node exposes neither .name nor .op_index.
    """
    resolved: List[_FullNode] = []

    for k, node in enumerate(dag):

        # ── op_index ──────────────────────────────────────────────────────
        if hasattr(node, 'op_index'):
            # Format C: already an integer index — use directly.
            op_index: int = int(node.op_index)

        elif hasattr(node, 'name'):
            # Formats A / B: map string name → integer index.
            op_name: str = node.name
            if op_name not in OP_NAME_TO_IDX:
                raise ValueError(
                    f"[shared_cnn] Unknown operation name '{op_name}' "
                    f"at DAG position {k}.\n"
                    f"Valid names: {list(OP_NAME_TO_IDX.keys())}\n"
                    f"Check that your Controller uses the same OP_NAMES List."
                )
            op_index = OP_NAME_TO_IDX[op_name]

        else:
            raise AttributeError(
                f"[shared_cnn] Node at position {k} exposes neither "
                f"'.name' (str) nor '.op_index' (int).\n"
                f"Received: {node!r}\n"
                f"Ensure your Controller builds Node objects from "
                f"utils.Node = namedtuple('Node', ['id', 'name'])."
            )

        # ── inputs (edge List) ────────────────────────────────────────────
        if hasattr(node, 'inputs') and node.inputs is not None:
            # Format B / C: explicit edge List provided.
            inputs: list[int] = list(node.inputs)
        else:
            # Format A: infer a linear chain topology.
            #   –1 is the sentinel for "stage input tensor".
            inputs = [-1] if k == 0 else [k - 1]

        resolved.append(_FullNode(id=int(node.id), op_index=op_index,
                                   inputs=inputs))

    return resolved


# ============================================================================
# SharedNode — one DAG node with all candidate ops pre-allocated
# ============================================================================

class SharedNode(nn.Module):
    """
    A single node in the searchable DAG.

    Holds one nn.Module per candidate operation so that all weights are
    allocated upfront and shared across every architecture sampled by the
    Controller (the ENAS weight-sharing principle).

    Args:
        node_id : position of this node in the DAG (0-based, for debugging).
        C       : channel depth (identical for all nodes within a stage).
    """

    def __init__(self, node_id: int, C: int):
        super().__init__()
        self.node_id = node_id
        # Pre-allocate every candidate operation. Only one is executed per
        # forward pass, but ALL weights are updated via backprop over many
        # different sampled architectures (parameter sharing).
        self.ops = nn.ModuleList([
            OP_BUILDERS[name](C, C) for name in OP_NAMES
        ])

    def forward(self, x: torch.Tensor, op_index: int) -> torch.Tensor:
        """
        Execute the single operation chosen by the Controller.

        Args:
            x        : input feature map  [B, C, H, W]
            op_index : integer index into OP_NAMES / self.ops

        Returns:
            Transformed feature map — same shape as x (all ops use same padding).
        """
        return self.ops[op_index](x)


# ============================================================================
# Channel calibration for skip connections
# ============================================================================

class _Calibration(nn.Module):
    """
    1×1 Conv + BN to project a skip tensor to a different channel count.
    Registered lazily through CNN.calibrations (an nn.ModuleDict) so that
    dynamically created projections still appear in model.parameters().
    """

    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ============================================================================
# Shared CNN child model
# ============================================================================

class CNN(nn.Module):
    """
    Shared Child CNN for ENAS Micro Search Space (CIFAR-10).

    Macro skeleton
    ──────────────
        Input [B, 3, 32, 32]
            │
           Stem  3×3 Conv → [B, C, 32, 32]
            │
          Stage 0  (num_blocks shared nodes)   → [B, C, 32, 32]
            │
          ↓ stride-2 downsample                → [B, C, 16, 16]
            │
          Stage 1  (num_blocks shared nodes)   → [B, C, 16, 16]
            │
          ↓ stride-2 downsample                → [B, C,  8,  8]
            │
          Stage 2  (num_blocks shared nodes)   → [B, C,  8,  8]
            │
          GAP  →  Linear(10)

    Args:
        args    : Namespace with at least:
                      args.cnn_hid   – base channel width  (e.g. 36)
                      args.num_blocks– nodes per stage DAG  (e.g. 6)
        dataset : dataset name string (kept for interface parity with RNN
                  shared models; only CIFAR-10 spatial dims are assumed).
    """

    NUM_STAGES:  int = 3    # fixed micro-search macro skeleton
    NUM_CLASSES: int = 10   # CIFAR-10

    def __init__(self, args, dataset: str):
        super().__init__()

        self.C           = args.cnn_hid
        self.num_blocks  = args.num_blocks
        self.dataset     = dataset

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
        )

        # ── Shared DAG nodes (one nn.ModuleList of SharedNodes per stage) ─
        self.stages = nn.ModuleList([
            nn.ModuleList([
                SharedNode(node_id=n, C=self.C)
                for n in range(self.num_blocks)
            ])
            for _ in range(self.NUM_STAGES)
        ])

        # ── Stride-2 transitions between stages ───────────────────────────
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.C, self.C, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.C),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.NUM_STAGES - 1)
        ])

        # ── Lazy calibration projections for skip connections ─────────────
        # Populated on demand by _get_calibration(); registered here so
        # that nn.Module tracks them properly for optimizer.parameters().
        self.calibrations = nn.ModuleDict()

        # ── Classifier ────────────────────────────────────────────────────
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.C, self.NUM_CLASSES)

        self._initialize_weights()

    # ── Weight initialisation ─────────────────────────────────────────────

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ── Lazy calibration helper ───────────────────────────────────────────

    def _get_calibration(self, key: str, C_in: int, C_out: int) -> nn.Module:
        """
        Return (and lazily create) a 1×1 calibration projection.
        Storing via self.calibrations ensures the weights are registered.
        """
        if key not in self.calibrations:
            device = next(self.parameters()).device
            self.calibrations[key] = _Calibration(C_in, C_out).to(device)
        return self.calibrations[key]

    # ── Trainer compatibility stub ────────────────────────────────────────

    def init_hidden(self, batch_size: int = None):
        """
        No recurrent state in a CNN. Returns None to satisfy the Trainer's
        unconditional call to model.init_hidden() at the start of each epoch.
        """
        return None

    # ── DAG traversal within one stage ────────────────────────────────────

    def _run_stage(
        self,
        stage_idx: int,
        x_in: torch.Tensor,
        dag: List,
    ) -> torch.Tensor:
        """
        Execute one DAG stage using the architecture sampled by the Controller.

        node_outputs[-1] = x_in   (sentinel for the stage-input tensor)
        node_outputs[k]  = output of DAG node k after applying its chosen op.

        Merging: when multiple predecessors feed a node, their tensors are
        summed element-wise (all ops preserve spatial dims via same-padding).

        Aggregation: the outputs of all *leaf* nodes (nodes never consumed as
        an input by a later node) are averaged.  This matches the ENAS paper
        and keeps gradients well-scaled regardless of DAG topology.

        Args:
            stage_idx : which stage (0/1/2) — selects self.stages[stage_idx].
            x_in      : stage input tensor  [B, C, H, W].
            dag       : List of _FullNode objects (already resolved by
                        _resolve_dag before this method is called).

        Returns:
            Stage output tensor [B, C, H, W].
        """
        shared_nodes = self.stages[stage_idx]
        C = self.C

        # –1 is the sentinel index for the stage-input tensor.
        node_outputs: Dict[int, torch.Tensor] = {-1: x_in}
        consumed: set[int] = set()   # tracks which node ids feed later nodes

        for node_desc in dag:
            nid     : int       = node_desc.id
            op_idx  : int       = node_desc.op_index
            inputs  : List[int] = node_desc.inputs

            # ── 1. Gather and merge predecessor tensors ──────────────────
            input_tensors: List[torch.Tensor] = []
            for src in inputs:
                consumed.add(src)
                t = node_outputs[src]
                if t.shape[1] != C:
                    # Defensive channel calibration (usually a no-op because
                    # all nodes share the same C throughout a stage).
                    cal_key = f"s{stage_idx}_src{src}_dst{nid}"
                    t = self._get_calibration(cal_key, t.shape[1], C)(t)
                input_tensors.append(t)

            if len(input_tensors) == 1:
                h = input_tensors[0]
            else:
                h = input_tensors[0]
                for t in input_tensors[1:]:
                    h = h + t           # element-wise addition

            # ── 2. Apply the selected operation (shared weights) ─────────
            node_outputs[nid] = shared_nodes[nid](h, op_idx)

        # ── 3. Average the outputs of leaf nodes ─────────────────────────
        leaf_ids = [
            nid for nid in node_outputs
            if nid != -1 and nid not in consumed
        ]

        if not leaf_ids:            # degenerate graph — fall back to last node
            leaf_ids = [dag[-1].id]

        leaf_tensors = [node_outputs[i] for i in leaf_ids]
        out = leaf_tensors[0]
        for t in leaf_tensors[1:]:
            out = out + t
        return out / len(leaf_tensors)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        dag,
        hidden=None,            # ignored; present for Trainer compatibility
    ) -> Tuple[torch.Tensor, None, Dict]:
        """
        Forward pass that handles both single DAGs (Shared training) and 
        batches of DAGs (Controller training).
        """
        
        # --- BATCH UNWRAPPING ---
        # During Controller training, dag is [[Node, Node...], [Node, Node...]]
        # We need to extract the specific DAG to evaluate.
        if isinstance(dag, list) and len(dag) > 0 and isinstance(dag[0], list):
            # If the first element is a list, it's a batch of DAGs.
            # We evaluate the first one in the batch.
            dag = dag[0]

        # --- DAG NORMALIZATION ---
        def _is_node(obj) -> bool:
            return hasattr(obj, 'id')

        # Heuristic: Determine if we have one DAG for all stages or unique DAGs per stage
        if (isinstance(dag, (list, tuple))
                and len(dag) == self.NUM_STAGES
                and not _is_node(dag[0])):
            # Format: [stage_0_dag, stage_1_dag, stage_2_dag]
            raw_stage_dags = list(dag)
        else:
            # Format: single flat dag — reuse for every stage.
            raw_stage_dags = [dag] * self.NUM_STAGES

        # --- RESOLVE NODES ---
        # Converts Node(id, name) -> internal _FullNode(id, op_index, inputs)
        stage_dags = [_resolve_dag(d) for d in raw_stage_dags]

        # --- COMPUTATION ---
        # 1. Stem
        h = self.stem(x)                                # [B, C, 32, 32]

        # 2. Stages
        for s in range(self.NUM_STAGES):
            h = self._run_stage(s, h, stage_dags[s])
            if s < self.NUM_STAGES - 1:
                h = self.downsamples[s](h)              # spatial resolution / 2

        # 3. Global Average Pooling & Classifier
        h = self.gap(h).view(h.size(0), -1)             # [B, C]
        logits = self.classifier(h)                     # [B, 10]

        # Return 3 values: (Output, Hidden, Extra_Info)
        return logits, None, {}

    # ── Debug repr ────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"dataset={self.dataset}, C={self.C}, "
            f"num_blocks={self.num_blocks}, num_stages={self.NUM_STAGES}, "
            f"ops={OP_NAMES}"
        )
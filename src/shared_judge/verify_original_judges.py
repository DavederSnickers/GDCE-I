"""Forward equivalence of the re-implemented ORIGINAL classifiers against the native code.

For one architecture, a randomly initialised re-implementation from shared_judge.py is
copied weight by weight into the native model of the baseline repo, and both are run on
real Mutagenicity test graphs: batched and one graph at a time, with unit edge weights
and with random soft edge weights in (0, 1) (the masks CF2, C2 and UCX optimise).

    c2gcn    C2's exp1_3 GCN, built verbatim on C2's own GCNConv (needs the C2 container)
    d4gcn    D4's gnns/mutag_gnn.py:Mutag_GCN with num_unit=2      (needs the D4 container)
    ucxgine  COMBINEX src/oracles/models/models.py:GINENet_G      (needs the COMBINEX container)
    cf2gcn   no DGL environment exists, so the native side is a dense transcription of
             DGL GraphConv(norm='both') on CF2's complete graph (utils/preprocessing/
             mutag_preprocessing_0.py) followed by gnn_cff/models/gcn.py:GCNGraph's head
    xplgcn   XPlore's src/oracle/nn/gcn.py:DownstreamGCN with its MUTAG settings, loaded by
             file path from the XPlore repo (--repo); needs only torch_geometric

    python verify_original_judges.py --arch c2gcn --repo ~/C2Explainer \
        --data_root ~/GraphDiffusionCounterfactualExplanation/data
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_judge import ORIGINAL_JUDGES, collate, load_split  # noqa: E402


def native_c2(repo, num_features):
    sys.path.insert(0, os.path.join(repo, "src"))
    from cf_explainer.gcn_conv import GCNConv
    from torch.nn import Linear
    from torch_geometric.nn import global_max_pool

    class GCN(torch.nn.Module):
        """exp/exp1_3.ipynb, cell 7, with the input width as an argument."""

        def __init__(self, nhid, nout, dropout):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_features, nhid, normalize=False)
            self.conv2 = GCNConv(nhid, nhid, normalize=False)
            self.conv3 = GCNConv(nhid, nout, normalize=False)
            self.lin = Linear(nout, 2)
            self.dropout = dropout

        def forward(self, x, edge_index, batch, edge_weight=None):
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_weight)
            x = x.relu()
            x = self.conv3(x, edge_index, edge_weight)
            x = global_max_pool(x, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin(x)
            return x

    return GCN(nhid=20, nout=20, dropout=0)


def copy_c2(ours, native):
    sd = {}
    for k in range(3):
        sd[f"conv{k + 1}.lin.weight"] = ours.lins[k].weight
        sd[f"conv{k + 1}.bias"] = ours.biases[k]
    sd["lin.weight"], sd["lin.bias"] = ours.lin.weight, ours.lin.bias
    native.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=True)


def run_c2(native, x, ei, batch, ea, w):
    return native(x, ei, batch, edge_weight=w)


def native_d4(repo, num_features):
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "gnns"))
    from gnns.mutag_gnn import Mutag_GCN
    return Mutag_GCN(2, feature_in=num_features)


def copy_d4(ours, native):
    sd = {k: v.detach().clone() for k, v in native.state_dict().items()}
    for k in range(2):
        sd[f"convs.{k}.lin.weight"] = ours.lins[k].weight.detach().clone()
        sd[f"convs.{k}.bias"] = ours.biases[k].detach().clone()
    sd["ffn.0.weight"], sd["ffn.0.bias"] = ours.ffn1.weight.detach().clone(), ours.ffn1.bias.detach().clone()
    sd["ffn.3.weight"], sd["ffn.3.bias"] = ours.ffn2.weight.detach().clone(), ours.ffn2.bias.detach().clone()
    native.load_state_dict(sd, strict=True)


def run_d4(native, x, ei, batch, ea, w):
    if w is None:
        return native(x, ei, batch)
    # get_pred_explain feeds sigmoid(mask * EPS) with EPS = 1 as edge weight
    return native.get_pred_explain(x, ei, torch.logit(w), batch)[1]


def native_ucx(repo, num_features):
    sys.path.insert(0, repo)
    from src.oracles.models.models import GINENet_G
    cfg = SimpleNamespace(model=SimpleNamespace(dropout=0.5, hidden_layers=[128, 128, 128]))
    return GINENet_G(num_features, 2, cfg, edge_attr_dim=5)


def copy_ucx(ours, native):
    sd = {k: v.detach().clone() for k, v in native.state_dict().items()}
    for i in (0, 2):
        sd[f"layers.0.nn.{i}.weight"] = ours.gine_mlp[i].weight.detach().clone()
        sd[f"layers.0.nn.{i}.bias"] = ours.gine_mlp[i].bias.detach().clone()
    sd["layers.0.lin.weight"] = ours.gine_lin.weight.detach().clone()
    sd["layers.0.lin.bias"] = ours.gine_lin.bias.detach().clone()
    for k in range(2):
        sd[f"layers.{k + 1}.lin.weight"] = ours.lins[k].weight.detach().clone()
        sd[f"layers.{k + 1}.bias"] = ours.biases[k].detach().clone()
    sd["output_layer.weight"] = ours.output_layer.weight.detach().clone()
    sd["output_layer.bias"] = ours.output_layer.bias.detach().clone()
    native.load_state_dict(sd, strict=True)


def run_ucx(native, x, ei, batch, ea, w):
    return native(x, ei, batch, edge_attr=ea, edge_weights=w)


def native_xpl(repo, num_features):
    """XPlore_config/GRAPH/MUTAG/MUTAG_GCN_XPlore.jsonc. The three model files are loaded by path;
    GRETEL's factory module (logger, config machinery) is replaced by the one function the GCN
    imports from it, whose non-string branch returns the pooling object unchanged."""
    import importlib.util
    import types
    for pkg in ("src", "src.core", "src.utils", "src.utils.torch", "src.oracle", "src.oracle.nn"):
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        sys.modules[pkg] = mod
    factory = types.ModuleType("src.core.factory_base")
    factory.build_w_params_string = lambda params: params
    sys.modules["src.core.factory_base"] = factory

    def load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(repo, rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("src.utils.norms", "src/utils/norms.py")
    load("src.utils.torch.gcn", "src/utils/torch/gcn.py")
    gcn = load("src.oracle.nn.gcn", "src/oracle/nn/gcn.py")
    return gcn.DownstreamGCN(num_features, n_classes=2, num_conv_layers=5, num_dense_layers=3,
                             conv_booster=2, linear_decay=1.8)


def copy_xpl(ours, native):
    sd = {k: v.detach().clone() for k, v in native.state_dict().items()}
    for k in range(ours.num_layers):
        p = f"graph_convs.{k}."
        sd[p + "norm.weight"] = ours.norm_weights[k].detach().clone()
        sd[p + "norm.bias"] = ours.norm_biases[k].detach().clone()
        sd[p + "conv.lin.weight"] = ours.lins[k].weight.detach().clone()
        sd[p + "conv.bias"] = ours.biases[k].detach().clone()
        for mine, theirs in ((ours.conv_res[k], "conv.residual"), (ours.layer_res[k], "res")):
            if isinstance(mine, torch.nn.Linear):
                sd[f"{p}{theirs}.weight"] = mine.weight.detach().clone()
                sd[f"{p}{theirs}.bias"] = mine.bias.detach().clone()
    for i, lin in enumerate(ours.dense):
        sd[f"downstream_layers.{2 * i}.weight"] = lin.weight.detach().clone()
        sd[f"downstream_layers.{2 * i}.bias"] = lin.bias.detach().clone()
    native.load_state_dict(sd, strict=True)


def run_xpl(native, x, ei, batch, ea, w):
    return native(x, ei, w, batch)


class DenseCF2(torch.nn.Module):
    """CF2's GCNGraph on the complete graph, one molecule at a time, transcribing DGL's
    GraphConv.forward for norm='both' with edge_weight (u_mul_e messages, sum reduce):
    feat_src * out_deg^-1/2, W before aggregation if in > out else after, rst * in_deg^-1/2,
    + bias. On the complete graph (incl. the (i, i) pairs, weight A_ii = 0) every
    structural degree is n. DGL stores W as (in, out)."""

    def __init__(self, ours):
        super().__init__()
        self.W = [ours.lins[k].weight.detach().t().clone() for k in range(3)]
        self.b = [ours.biases[k].detach().clone() for k in range(3)]
        self.dense = [ours.dense1, ours.dense2, ours.dense3]

    def forward(self, x, ei, batch, ea, w):
        outs = []
        for g in range(int(batch.max()) + 1):
            nodes = (batch == g).nonzero().view(-1)
            n, off = len(nodes), int(nodes[0])
            sel = (ei[0] >= off) & (ei[0] < off + n)
            A = torch.zeros(n, n, dtype=x.dtype)  # A[u, v] = weight of message u -> v
            A[ei[0, sel] - off, ei[1, sel] - off] = 1.0 if w is None else w[sel]
            h = x[nodes]
            for W, b in zip(self.W, self.b):
                feat_src = h * n ** -0.5
                if W.shape[0] > W.shape[1]:
                    rst = A.t() @ (feat_src @ W)
                else:
                    rst = (A.t() @ feat_src) @ W
                h = torch.relu(rst * n ** -0.5 + b)
            z = h.mean(dim=0, keepdim=True)
            z = torch.relu(self.dense[0](z))
            z = torch.relu(self.dense[1](z))
            outs.append(torch.sigmoid(self.dense[2](z)))
        return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ORIGINAL_JUDGES))
    ap.add_argument("--repo", default="", help="baseline repo root (not needed for cf2gcn)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset", default="mutagenicity_mol")
    ap.add_argument("--n_graphs", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(0)
    graphs, n_feat, n_edge = load_split(args.data_root, args.dataset, "test")
    graphs = graphs[: args.n_graphs]
    ours = ORIGINAL_JUDGES[args.arch](n_feat, 2, edge_dim=n_edge).double().eval()
    with torch.no_grad():  # random, non-zero biases so that a misplaced bias cannot hide
        for p in ours.parameters():
            p.add_(0.1 * torch.randn_like(p))

    if args.arch == "c2gcn":
        native = native_c2(args.repo, n_feat)
        copy_c2(ours, native)
        run = run_c2
    elif args.arch == "d4gcn":
        native = native_d4(args.repo, n_feat)
        copy_d4(ours, native)
        run = run_d4
    elif args.arch == "ucxgine":
        native = native_ucx(args.repo, n_feat)
        copy_ucx(ours, native)
        run = run_ucx
    elif args.arch == "xplgcn":
        native = native_xpl(args.repo, n_feat)
        copy_xpl(ours, native)
        run = run_xpl
    else:
        native = DenseCF2(ours)
        run = lambda model, *inputs: model(*inputs)  # noqa: E731
    native = native.double().eval()

    def ours_out(x, ei, batch, ea, w):
        logits = ours(x, ei, batch, edge_attr=ea, edge_weights=w)
        if args.arch == "ucxgine":
            return F.log_softmax(logits, dim=1)  # GINENet_G returns log_softmax
        if args.arch == "cf2gcn":
            return torch.softmax(logits, dim=1)[:, 1:]  # GCNGraph returns sigmoid(z)
        return logits

    worst = 0.0
    agree = total = 0
    gen = torch.Generator().manual_seed(1)
    for mode in ("batched", "single"):
        chunks = [graphs] if mode == "batched" else [[g] for g in graphs]
        for chunk in chunks:
            x, ei, batch, ea, _ = collate(chunk)
            x, ea = x.double(), (ea.double() if ea is not None else None)
            for soft in (False, True):
                w = (torch.rand(ei.shape[1], generator=gen, dtype=torch.float64) * 0.98 + 0.01
                     if soft else None)
                with torch.no_grad():
                    a = ours_out(x, ei, batch, ea, w)
                    b = run(native, x, ei, batch, ea, w)
                worst = max(worst, float((a - b).abs().max()))
                if args.arch == "cf2gcn":
                    agree += int(((a > 0.5) == (b > 0.5)).sum())
                else:
                    agree += int((a.argmax(1) == b.argmax(1)).sum())
                total += a.shape[0]
    print(f"[verify] arch={args.arch} graphs={len(graphs)} torch={torch.__version__} "
          f"max|ours - native|={worst:.3e} prediction agreement={agree}/{total}")
    ok = worst < args.tol and agree == total
    print("[verify] PASS" if ok else "[verify] FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

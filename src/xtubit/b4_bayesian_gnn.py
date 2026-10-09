from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GINEConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

    class GINEConv(nn.Module):
        """Fail-closed placeholder when torch_geometric is not installed."""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError(
                "PyTorch Geometric (torch_geometric) is required for authentic GINEConv Bayesian GNN execution. "
                "To preserve scientific validity and prevent silent architectural substitution, execution fails closed. "
                "Install torch_geometric or use the production Bayesian descriptor surrogate (run_stage_b3_b4)."
            )

    def global_mean_pool(*args, **kwargs):
        """Fail-closed placeholder when torch_geometric is not installed."""
        raise ImportError(
            "PyTorch Geometric (torch_geometric) is required for global_mean_pool. Execution fails closed."
        )


class BayesianLinear(nn.Module):
    """Diagonal Gaussian variational linear layer.

    The layer samples on every forward, including evaluation, so repeated
    evaluation passes produce a posterior predictive sample for MC inference.
    """
    def __init__(self, in_features, out_features, prior_sigma=0.1):
        super().__init__()
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.02))
        self.rho_w = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.mu_b = nn.Parameter(torch.zeros(out_features))
        self.rho_b = nn.Parameter(torch.full((out_features,), -5.0))
        self.prior_sigma = float(prior_sigma)

    @staticmethod
    def _sigma(rho):
        return F.softplus(rho) + 1e-8

    def forward(self, x):
        sw = self._sigma(self.rho_w)
        sb = self._sigma(self.rho_b)
        w = self.mu_w + sw * torch.randn_like(self.mu_w)
        b = self.mu_b + sb * torch.randn_like(self.mu_b)
        return F.linear(x, w, b)

    def kl(self):
        ps = self.prior_sigma
        sw = self._sigma(self.rho_w)
        sb = self._sigma(self.rho_b)
        prior_var = ps * ps
        kl_w = torch.log(torch.tensor(ps, device=sw.device) / sw) + (sw.pow(2) + self.mu_w.pow(2)) / (2*prior_var) - 0.5
        kl_b = torch.log(torch.tensor(ps, device=sb.device) / sb) + (sb.pow(2) + self.mu_b.pow(2)) / (2*prior_var) - 0.5
        return kl_w.sum() + kl_b.sum()


class BayesianGNN(nn.Module):
    def __init__(self, atom_dim, edge_dim, fa_dim=0, hidden=128, layers=3, prior_sigma=0.1, use_fa=True):
        super().__init__()
        self.use_fa = bool(use_fa)
        self.in_dim = atom_dim + (fa_dim if self.use_fa else 0)
        self.input = nn.Linear(self.in_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
        self.head = nn.Sequential(
            BayesianLinear(hidden, hidden, prior_sigma), nn.SiLU(),
            BayesianLinear(hidden, 1, prior_sigma),
        )

    def forward(self, batch):
        x = batch.x
        if self.use_fa:
            x = torch.cat([x, batch.h_faenet], dim=-1)
        x = self.input(x)
        for conv in self.convs:
            x = F.silu(conv(x, batch.edge_index, batch.edge_attr) + x)
        x = global_mean_pool(x, batch.batch)
        return self.head(x).squeeze(-1)

    def kl(self):
        return sum(m.kl() for m in self.modules() if isinstance(m, BayesianLinear))


def loss(model, batch, target, kl_weight=1e-2, target_scale=1.0):
    pred = model(batch)
    mse = F.mse_loss(pred, target)
    return mse + kl_weight * model.kl() / max(1, target.shape[0]) / max(target_scale, 1e-8)


@torch.no_grad()
def mc_predict(model, loader, samples=50, device="cuda"):
    model.eval()
    all_samples = []
    for _ in range(samples):
        vals=[]
        for batch in loader:
            batch=batch.to(device)
            vals.append(model(batch).detach().cpu())
        all_samples.append(torch.cat(vals))
    y=torch.stack(all_samples,0)
    return y.mean(0), y.std(0, unbiased=True)


def calibrate_sigma(mu, sigma, y, min_tau=1e-3, max_tau=100.0):
    """Fit a scalar sigma multiplier by one-dimensional log-grid NLL search."""
    device=mu.device
    grid=torch.logspace(math.log10(min_tau), math.log10(max_tau), 400, device=device)
    err=(y-mu).pow(2)
    nll=[]
    for tau in grid:
        ss=torch.clamp(sigma*tau, min=1e-6)
        nll.append(0.5*(torch.log(ss.pow(2))+err/ss.pow(2)).mean())
    tau=grid[int(torch.argmin(torch.stack(nll)))]
    return float(tau)

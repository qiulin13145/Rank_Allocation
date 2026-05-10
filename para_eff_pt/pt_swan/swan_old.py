import math
import warnings
from typing import Iterable, Callable, Optional
import torch
from torch.optim import Optimizer

@torch.compile
def newtonschulz(Y, Z, whiten_iters, diag, beta, I):
    # Newton-Schulz iteration to approximate S^{-1/2}
    for _ in range(whiten_iters):
        if diag:
            # Diagonal substitution scheme (SWAN⧧)
            Y_diag = torch.diag(Y)
            Z_diag = torch.diag(Z)
            
            # Y ← β · Y · Diag(3I - Z · Diag(Y))
            Y = beta * torch.matmul(
                Y, 
                torch.diag(3 * torch.ones_like(Y_diag) - Z_diag * Y_diag)
            )
            
            # Z ← β · (3I - Diag(Z) · Y) · Diag(Z)
            Z = beta * torch.matmul(
                torch.diag(3 * torch.ones_like(Z_diag) - Z_diag * Y_diag),
                torch.diag(Z_diag)
            )
        else:
            # Standard Newton-Schulz scheme
            # Y ← β · Y · (3I - Z · Y)
            Y = beta * torch.matmul(Y, (3 * I - torch.matmul(Z, Y)))
            # Z ← β · (3I - Z · Y) · Z
            Z = beta * torch.matmul((3 * I - torch.matmul(Z, Y)), Z)
            
    return Z



class SWAN(Optimizer):
    r"""Implements SWAN optimizer that preprocesses the instantaneous gradients using
    GradNorm and GradWhitening operators for 2D linear layers, while using AdamW for other layers.

    SWAN is designed for large-scale training. For 2D parameters (linear layers), it:
    1. Normalizes the gradient rows (GradNorm)
    2. Whitens the normalized gradient using a Newton-Schulz iterative method (GradWhitening)
    3. Optionally normalizes the update direction to match √(m*n)/||·||
    4. Applies a scaled learning rate (lr * alpha)

    For other parameters (1D or 3D+), it uses standard AdamW optimizer with the original learning rate.

    Arguments:
        params (Iterable): Iterable of parameters
        lr (float, optional): Learning rate (default: 1e-3)
        alpha (float, optional): Scaling factor for SWAN learning rate on linear layers (default: 0.05)
        betas (tuple, optional): Coefficients for computing running averages in AdamW (default: (0.9, 0.999))
        whiten_iters (int, optional): Number of Newton-Schulz iterations for whitening (default: 2)
        beta (float, optional): Coefficient for Newton-Schulz iteration (default: 0.4)
        weight_decay (float, optional): Weight decay factor for AdamW (default: 0.01)
        diag (bool, optional): If True, use diagonal substitution scheme (default: True)
        normalize_update (bool, optional): If True, normalize update direction (default: True)
        eps (float, optional): Term for numerical stability (default: 1e-8)

    Example:
        >>> optimizer = SWAN(model.parameters(), lr=1e-3, alpha=0.05)
    """
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        alpha: float = 0.05,
        betas: tuple = (0.9, 0.999),
        whiten_iters: int = 2,
        beta: float = 0.4,
        weight_decay: float = 0.00,
        diag: bool = True,
        normalize_update: bool = True,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if alpha <= 0.0:
            raise ValueError(f"Invalid alpha value: {alpha} - should be > 0.0")
        if whiten_iters < 1:
            raise ValueError(f"Invalid whiten_iters: {whiten_iters} - should be at least 1")
        if beta <= 0.0 or beta > 1.0:
            raise ValueError(f"Invalid beta value: {beta} - should be in (0, 1]")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = {
            "lr": lr,
            "alpha": alpha,
            "betas": betas,
            "whiten_iters": whiten_iters,
            "beta": beta,
            "weight_decay": weight_decay,
            "diag": diag,
            "normalize_update": normalize_update,
            "eps": eps,
        }
        
        # Initialize optimizer with provided parameters
        super().__init__(params, defaults)
        
        # Initialize AdamW state for all parameters
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.

        Arguments:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over parameter groups
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            betas = group["betas"]
            whiten_iters = group["whiten_iters"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            diag = group["diag"]
            normalize_update = group["normalize_update"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError("SWAN does not support sparse gradients.")
                
                # Get optimizer state
                state = self.state[p]
                state['step'] += 1
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Determine optimization method based on parameter dimensionality
                if grad.dim() == 2:
                    # ----- SWAN optimizer for 2D parameters (linear layers) -----
                    
                    # Apply SWAN with scaled learning rate
                    grad_processed = self._apply_swan(grad, whiten_iters, beta, diag, normalize_update, eps)
                    
                    # Use scaled learning rate (lr * alpha) for SWAN
                    effective_lr = lr * alpha
                    p.add_(grad_processed, alpha=-effective_lr)
                    
                else:
                    # ----- AdamW optimizer for non-2D parameters -----
                    
                    # Get optimizer state
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = betas
                    
                    # Decay the first and second moment running average coefficients
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    # Compute adaptive learning rate (using original lr)
                    step_size = lr / bias_correction1
                    
                    # Compute Adam update
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    update = exp_avg / denom
                    
                    # Update parameter with original learning rate
                    p.add_(update, alpha=-step_size)

        return loss
    
    def _apply_swan(self, grad, whiten_iters, beta, diag, normalize_update, eps):
        """
        Apply SWAN optimization steps to the gradient.
        
        Args:
            grad: The gradient tensor (should be 2D).
            whiten_iters: Number of Newton-Schulz iterations.
            beta: Coefficient for Newton-Schulz iteration.
            diag: Whether to use diagonal substitution scheme.
            normalize_update: Whether to normalize update direction to √(m*n)/||·||.
            eps: Small constant for numerical stability.
            
        Returns:
            Processed gradient tensor.
        """
        # --- Step 1. GradNorm: Row-wise gradient normalization ---
        # Compute the RMS for each row (RMS)
        # s shape: (m, 1)
        s = grad.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=eps)
        grad_norm = grad / s

        # --- Step 2. GradWhitening: Whitening the normalized gradient ---
        # Compute the Gram matrix S = grad_norm * grad_norm^T; shape of S is (m, m)
        S = torch.matmul(grad_norm, grad_norm.t())

        # Initialize Newton-Schulz iteration variables
        Y = S.clone()
        m, n = grad.size()  # get dimensions of the matrix
        I = torch.eye(m, device=grad.device, dtype=grad.dtype)
        Z = I.clone()

        # Newton-Schulz iteration to approximate S^{-1/2}
        '''
        for _ in range(whiten_iters):
            if diag:
                # Diagonal substitution scheme (SWAN⧧)
                Y_diag = torch.diag(Y)
                Z_diag = torch.diag(Z)
                
                # Y ← β · Y · Diag(3I - Z · Diag(Y))
                Y = beta * torch.matmul(
                    Y, 
                    torch.diag(3 * torch.ones_like(Y_diag) - Z_diag * Y_diag)
                )
                
                # Z ← β · (3I - Diag(Z) · Y) · Diag(Z)
                Z = beta * torch.matmul(
                    torch.diag(3 * torch.ones_like(Z_diag) - Z_diag * Y_diag),
                    torch.diag(Z_diag)
                )
            else:
                # Standard Newton-Schulz scheme
                # Y ← β · Y · (3I - Z · Y)
                Y = beta * torch.matmul(Y, (3 * I - torch.matmul(Z, Y)))
                # Z ← β · (3I - Z · Y) · Z
                Z = beta * torch.matmul((3 * I - torch.matmul(Z, Y)), Z)
        '''
        
        Z = newtonschulz(Y, Z, whiten_iters, diag, beta, I)

        # Whitening the gradient: grad_white = Z * grad_norm
        grad_white = torch.matmul(Z, grad_norm)

        # --- Step 3. Optional Normalization ---
        # According to the paper: ΔW^(t) ← (√(m*n) ΔW^(t)) / ||ΔW^(t)||
        if normalize_update:
            norm_grad_white = grad_white.norm()
            if norm_grad_white > eps:
                sqrt_mn = (m * n) ** 0.5
                grad_white = sqrt_mn * grad_white / norm_grad_white
        
        return grad_white
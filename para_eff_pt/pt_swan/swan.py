import os
import math
import torch
import torch.distributed as dist


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() # ???????????????????
    
    #X = G # .bfloat16()
    
    #print("X dtype = ", X.dtype)
    
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class SWAN(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        swan_version="v2",
        muon_params=None,
        momentum=0.9,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        
        self.swan_version = swan_version
        
 
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"] # not used when doing bias correction 
            wd = group["wd"]
            beta = group["momentum"]
            
            # generate weight updates in distributed fashion
            for p in params:    
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None
 
                # calc update
                state = self.state[p]

                # all row-wise are col-wise for linear layers and the same for col-wise ....
                # dont do anything with state[step] !!

                if self.swan_version == "v2":
                
                    if "step" not in state:
                        state["step"] = 0
                        state["row_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=1, keepdim=True)) 

                    row_sq_avg = state["row_sq_avg"]
                    state["row_sq_avg"] =  beta * row_sq_avg + (1 - beta) * g.pow(2).mean(dim=1, keepdim=True)   
                    row_sq_avg = state["row_sq_avg"] 
                    
                    #print(row_sq_avg.shape)
                    
                    s = row_sq_avg.sqrt().clamp(min=1e-8)
                    u = g / s
                
                elif self.swan_version == "v1":
                    s = g.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
                    u = g / s

                    #lr = group["lr"]
                    #lr*= 1.0 #0.05
                
                elif self.swan_version == 'v2_column':
                    if "step" not in state:
                        state["step"] = 0
                        state["col_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=0, keepdim=True)) 
                        
                    col_sq_avg = state["col_sq_avg"]
                    state["col_sq_avg"] =  beta * col_sq_avg + (1 - beta) * g.pow(2).mean(dim=0, keepdim=True)   
                    col_sq_avg = state["col_sq_avg"] 
                    
                    s = col_sq_avg.sqrt().clamp(min=1e-8)
                    u = g / s
                
                elif self.swan_version == 'v2_max':
                                        
                    m,n = p.shape[0], p.shape[1]   # not always true !!! not for Linear which use transpose !!!
                    
                    #print(m,n)
                    
                    
                    if m >= n: #  ???
                        
                        #print(m,n)
                        
                        # ROW-WISE
                        if "step" not in state:
                            state["step"] = 0
                            state["row_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=1, keepdim=True)) 

                        row_sq_avg = state["row_sq_avg"]
                        state["row_sq_avg"] =  beta * row_sq_avg + (1 - beta) * g.pow(2).mean(dim=1, keepdim=True)   
                        row_sq_avg = state["row_sq_avg"] 
                        
                        #print(row_sq_avg.shape)
                        
                        s = row_sq_avg.sqrt().clamp(min=1e-8)
                        u = g / s     
                    else: # m < n ???  
                        #print(m,n)
                        # COLUMN-WISE
                        if "step" not in state:
                            state["step"] = 0
                            state["col_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=0, keepdim=True)) 
                            
                        col_sq_avg = state["col_sq_avg"]
                        state["col_sq_avg"] =  beta * col_sq_avg + (1 - beta) * g.pow(2).mean(dim=0, keepdim=True)   
                        col_sq_avg = state["col_sq_avg"] 
                        
                        s = col_sq_avg.sqrt().clamp(min=1e-8)
                        u = g / s                        
                                      
                elif self.swan_version == 'v2_both':
                                        
                    m,n = p.shape[0], p.shape[1]   # not always true !!! not for Linear which use transpose !!!
                    
                    if m != 32000 and n != 32000 : # ???
                        
                        #print(p.shape)
                        #BOTH ??
                        if "row_sq_avg" not in state:
                            state["step"] = 0
                            state["row_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=1, keepdim=True)) 

                        row_sq_avg = state["row_sq_avg"]
                        state["row_sq_avg"] =  beta * row_sq_avg + (1 - beta) * g.pow(2).mean(dim=1, keepdim=True)   
                        row_sq_avg = state["row_sq_avg"] 
                        
                        if "col_sq_avg" not in state:
                            #state["step"] = 0
                            state["col_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=0, keepdim=True)) 
                            
                        col_sq_avg = state["col_sq_avg"]
                        state["col_sq_avg"] =  beta * col_sq_avg + (1 - beta) * g.pow(2).mean(dim=0, keepdim=True)   
                        col_sq_avg = state["col_sq_avg"] 
                        
                        
                        c = col_sq_avg.sqrt().clamp(min=1e-8)
                        s = row_sq_avg.sqrt().clamp(min=1e-8)
                        
                        
                        
                        u = g / s
                        u = u / c
 
                    else:
                        # ROW-WISE
                        if "step" not in state:
                            state["step"] = 0
                            state["row_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=1, keepdim=True)) 

                        row_sq_avg = state["row_sq_avg"]
                        state["row_sq_avg"] =  beta * row_sq_avg + (1 - beta) * g.pow(2).mean(dim=1, keepdim=True)   
                        row_sq_avg = state["row_sq_avg"] 
                        
                        #print(row_sq_avg.shape)
                        
                        s = row_sq_avg.sqrt().clamp(min=1e-8)
                        u = g / s    
                        
                      
                elif self.swan_version == 'v2_both_all':
                                        
                    m,n = p.shape[0], p.shape[1]   # not always true !!! not for Linear which use transpose !!!
                    
                    #if m != 32000 and n != 32000 : # ???
                        
                    #print(p.shape)
                    #BOTH ??
                    if "row_sq_avg" not in state:
                        state["step"] = 0
                        state["row_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=1, keepdim=True)) 

                    row_sq_avg = state["row_sq_avg"]
                    state["row_sq_avg"] =  beta * row_sq_avg + (1 - beta) * g.pow(2).mean(dim=1, keepdim=True)   
                    row_sq_avg = state["row_sq_avg"] 
                    
                    if "col_sq_avg" not in state:
                        #state["step"] = 0
                        state["col_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=0, keepdim=True)) 
                        
                    col_sq_avg = state["col_sq_avg"]
                    state["col_sq_avg"] =  beta * col_sq_avg + (1 - beta) * g.pow(2).mean(dim=0, keepdim=True)   
                    col_sq_avg = state["col_sq_avg"] 
                    
                    
                    c = col_sq_avg.sqrt().clamp(min=1e-8)
                    s = row_sq_avg.sqrt().clamp(min=1e-8)
                    
                    
                    
                    u = g / s
                    u = u / c
 
                      
                        
                elif self.swan_version == 'v2_both_4th':
                                        
                    m,n = p.shape[0], p.shape[1]   # not always true !!! not for Linear which use transpose !!!
                    
                    if m != 32000 and n != 32000 : # ???
                        
                        #print(p.shape)
                        #BOTH ??
                        if "row_sq_avg" not in state:
                            state["step"] = 0
                            state["row_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=1, keepdim=True)) 

                        row_sq_avg = state["row_sq_avg"]
                        state["row_sq_avg"] =  beta * row_sq_avg + (1 - beta) * g.pow(2).mean(dim=1, keepdim=True)   
                        row_sq_avg = state["row_sq_avg"] 
                        
                        if "col_sq_avg" not in state:
                            #state["step"] = 0
                            state["col_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=0, keepdim=True)) 
                            
                        col_sq_avg = state["col_sq_avg"]
                        state["col_sq_avg"] =  beta * col_sq_avg + (1 - beta) * g.pow(2).mean(dim=0, keepdim=True)   
                        col_sq_avg = state["col_sq_avg"] 
                        
                        
                        c = col_sq_avg.sqrt().sqrt().clamp(min=1e-8)   # 1/4 !!
                        s = row_sq_avg.sqrt().sqrt().clamp(min=1e-8)   # 1/4 !!
                        
                        
                        
                        u = g / s
                        u = u / c
                        
                                                                        
                    else:
                        # ROW-WISE
                        if "step" not in state:
                            state["step"] = 0
                            state["row_sq_avg"] = torch.zeros_like(g.pow(2).mean(dim=1, keepdim=True)) 

                        row_sq_avg = state["row_sq_avg"]
                        state["row_sq_avg"] =  beta * row_sq_avg + (1 - beta) * g.pow(2).mean(dim=1, keepdim=True)   
                        row_sq_avg = state["row_sq_avg"] 
                        
                        #print(row_sq_avg.shape)
                        
                        s = row_sq_avg.sqrt().clamp(min=1e-8)
                        u = g / s     
                    
                    
                    
                    
                
                elif self.swan_version == 'no_gradnorm':
                    u = g                  
                    
                else:
                    raise ValueError(f"invalid choice for swan_version argmument...")     






                u = zeropower_via_newtonschulz5(u, steps=group["ns_steps"])
 
                    
                # do it for swan or not ??    
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)  

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
 
            for p in params:          
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss

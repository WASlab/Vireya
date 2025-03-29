
####################################################################################################
# Muon - MomentUm Orthogonalized by Newton-schulz
# Copied From: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
# Original: https://github.com/KellerJordan/Muon
####################################################################################################
import torch
import math
import warnings
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
    X = G.bfloat16()
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


class Muon(torch.optim.Optimizer):
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
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
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
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

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
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
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
    
class StepLRScheduler:
    """
    A step-wise learning rate scheduler that updates on *every batch* (each training step).
    
    Features:
      - Optional warmup from init_lr to peak_lr (over warmup_steps).
      - Optional decay from peak_lr to end_lr (over total_steps - warmup_steps).
      - Multiple warmup/decay types: "linear", "exponential", "polynomial", "cosine", "sigmoid", "static", and (for decay) "logarithmic".
      - If init_lr is False, the warmup phase is skipped (immediately using peak_lr).
      - If end_lr is False, the decay phase is skipped (staying at peak_lr after warmup).
      
    Usage:
      1) Initialize with an optimizer, total_steps, warmup_steps, etc.
      2) Call .step() each time you process a batch (i.e., each training step).
      3) The scheduler automatically updates optimizer.param_groups[...]["lr"].
    """

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=0,
        init_lr=1e-4,
        peak_lr=1e-3,
        end_lr=1e-5,
        warmup_type="linear",
        decay_type="cosine",
    ):
        """
        Args:
            optimizer: torch Optimizer object.
            total_steps (int): total number of steps (batches) for training.
            warmup_steps (int): number of steps used for warmup.
            init_lr (float or bool): starting LR. If False, no warmup phase (start at peak_lr).
            peak_lr (float): LR at the end of warmup. If no warmup, remains constant until decay starts.
            end_lr (float or bool): final LR after decay. If False, no decay occurs; remain at peak_lr.
            warmup_type (str): "linear", "exponential", "polynomial", "cosine", "sigmoid", or "static".
            decay_type (str): "linear", "exponential", "polynomial", "cosine", "sigmoid", "logarithmic", or "static".
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_type = warmup_type.lower()
        self.decay_type = decay_type.lower()

        self.current_step = 0

        # Validate arguments
        if self.init_lr is False and self.warmup_steps > 0:
            warnings.warn(
                "warmup_steps > 0 but init_lr=False. Warmup phase is disabled; training starts at peak_lr."
            )
        if self.end_lr is False and (self.decay_type != "static"):
            warnings.warn(
                "end_lr=False but decay_type is not 'static'. Decay phase is disabled; LR stays at peak_lr."
            )
        if self.total_steps <= self.warmup_steps and self.warmup_steps > 0:
            warnings.warn(
                f"total_steps={self.total_steps} is less than or equal to warmup_steps={self.warmup_steps}. "
                "Decay phase will not occur."
            )

        # Initialize optimizer with starting LR
        initial = self.init_lr if self.init_lr is not False else self.peak_lr
        self._set_lr(initial)

    def step(self):
        """Call this once per training step (batch)."""
        self.current_step += 1
        new_lr = self._compute_lr()
        self._set_lr(new_lr)

    def _compute_lr(self):
        """Compute the learning rate based on the current step."""
        if (self.init_lr is False) and (self.end_lr is False):
            return self.peak_lr

        if self.init_lr is not False and (self.current_step <= self.warmup_steps):
            return self._warmup_lr()
        else:
            if self.end_lr is False:
                return self.peak_lr
            return self._decay_lr()

    def _warmup_lr(self):
        """Compute LR during the warmup phase."""
        progress = self.current_step / max(1, self.warmup_steps)  # in [0,1]
        if self.warmup_type == "linear":
            return self.init_lr + (self.peak_lr - self.init_lr) * progress
        elif self.warmup_type == "exponential":
            ratio = self.peak_lr / max(1e-12, self.init_lr)
            return self.init_lr * (ratio ** progress)
        elif self.warmup_type == "polynomial":
            return self.init_lr + (self.peak_lr - self.init_lr) * (progress ** 2)
        elif self.warmup_type == "static":
            return self.init_lr
        elif self.warmup_type == "cosine":
            # Cosine warmup: f(0)=0, f(1)=1 using 1-cos((pi/2)*progress)
            return self.init_lr + (self.peak_lr - self.init_lr) * (1 - math.cos((math.pi * progress) / 2))
        elif self.warmup_type == "sigmoid":
            # Sigmoid warmup: Normalize a sigmoid so that f(0)=0 and f(1)=1.
            k = 12.0  # steepness constant
            raw0 = 1 / (1 + math.exp(k * 0.5))
            raw1 = 1 / (1 + math.exp(-k * 0.5))
            raw = 1 / (1 + math.exp(-k * (progress - 0.5)))
            norm_factor = (raw - raw0) / (raw1 - raw0)
            return self.init_lr + (self.peak_lr - self.init_lr) * norm_factor
        else:
            raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

    def _decay_lr(self):
        """Compute LR during the decay phase."""
        decay_step = self.current_step - self.warmup_steps
        decay_total = max(1, self.total_steps - self.warmup_steps)
        progress = decay_step / decay_total  # in [0,1]
        if self.decay_type == "linear":
            return self.peak_lr + (self.end_lr - self.peak_lr) * progress
        elif self.decay_type == "exponential":
            ratio = self.end_lr / max(1e-12, self.peak_lr)
            return self.peak_lr * (ratio ** progress)
        elif self.decay_type == "polynomial":
            return self.peak_lr + (self.end_lr - self.peak_lr) * (progress ** 2)
        elif self.decay_type == "cosine":
            return self.end_lr + 0.5 * (self.peak_lr - self.end_lr) * (1 + math.cos(math.pi * progress))
        elif self.decay_type == "static":
            return self.peak_lr
        elif self.decay_type == "sigmoid":
            # Sigmoid decay: Normalize a sigmoid so that f(0)=0 and f(1)=1.
            k = 12.0
            raw0 = 1 / (1 + math.exp(-k * 0.5))
            raw1 = 1 / (1 + math.exp(k * 0.5))
            raw = 1 / (1 + math.exp(k * (progress - 0.5)))
            norm_factor = (raw - raw0) / (raw1 - raw0)
            return self.peak_lr + (self.end_lr - self.peak_lr) * norm_factor
        elif self.decay_type == "logarithmic":
            # Logarithmic decay: f(progress)=1 - log(1+progress*(e-1)) (normalized so that f(0)=1 and f(1)=0)
            factor = 1 - math.log(1 + progress * (math.e - 1))
            return self.peak_lr * factor + self.end_lr * (1 - factor)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")

    def _set_lr(self, lr):
        """Sets the learning rate in all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Returns the current LR from the first parameter group."""
        return self.optimizer.param_groups[0]["lr"]
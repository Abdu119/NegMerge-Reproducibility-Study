# Referenced from: https://github.com/gortizji/tangent_task_arithmetic/blob/main/src/utils.py

import os
import pickle
 
import numpy as np
import torch


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)


def patch_vision_transformer_deep(model, verbose=False):
    """
    Deeply patch ALL VisionTransformer modules in a model hierarchy.
    
    This handles the case where VisionTransformer is nested inside:
    - ImageEncoder.model.visual
    - Any other nested structure
    
    Must be called AFTER torch.load() to ensure all modules are instantiated.
    """
    patched_count = 0
    
    # Method 1: Use named_modules() to find ALL modules at any depth
    if hasattr(model, 'named_modules'):
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'VisionTransformer':
                if not hasattr(module, 'global_average_pool'):
                    module.global_average_pool = False
                    patched_count += 1
                    if verbose:
                        print(f"  Patched global_average_pool on: {name}")
                if not hasattr(module, 'output_tokens'):
                    module.output_tokens = False
                    if verbose:
                        print(f"  Patched output_tokens on: {name}")
    
    # Method 2: Also check common locations directly
    # For ImageEncoder structure: model.model.visual
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        visual = model.model.visual
        if visual.__class__.__name__ == 'VisionTransformer':
            if not hasattr(visual, 'global_average_pool'):
                visual.global_average_pool = False
                patched_count += 1
                if verbose:
                    print(f"  Patched model.model.visual.global_average_pool")
            if not hasattr(visual, 'output_tokens'):
                visual.output_tokens = False
                if verbose:
                    print(f"  Patched model.model.visual.output_tokens")
    
    # For direct CLIP model structure: model.visual
    if hasattr(model, 'visual'):
        visual = model.visual
        if visual.__class__.__name__ == 'VisionTransformer':
            if not hasattr(visual, 'global_average_pool'):
                visual.global_average_pool = False
                patched_count += 1
                if verbose:
                    print(f"  Patched model.visual.global_average_pool")
            if not hasattr(visual, 'output_tokens'):
                visual.output_tokens = False
                if verbose:
                    print(f"  Patched model.visual.output_tokens")
    
    return patched_count


def patch_vision_transformer(module):
    """Recursively patch VisionTransformer modules with missing attributes.
    
    DEPRECATED: Use patch_vision_transformer_deep() instead for more reliable patching.
    """
    # Check if this module is a VisionTransformer (by class name to avoid import issues)
    if module.__class__.__name__ == 'VisionTransformer':
        if not hasattr(module, 'global_average_pool'):
            module.global_average_pool = False
        if not hasattr(module, 'output_tokens'):
            module.output_tokens = False
    
    # Recursively patch children
    for child in module.children():
        patch_vision_transformer(child)


def torch_load(save_path, device=None, verbose=False):
    """
    Load a PyTorch checkpoint with proper handling for older checkpoints.
    
    This function:
    1. Loads with weights_only=False for PyTorch 2.6+ compatibility
    2. Deeply patches any VisionTransformer modules for open_clip compatibility
    """
    # weights_only=False needed for PyTorch 2.6+ to load custom classes
    model = torch.load(save_path, map_location="cpu", weights_only=False)
    
    # CRITICAL: Patch VisionTransformer modules AFTER loading
    # This handles checkpoints saved with older open_clip versions
    patched = patch_vision_transformer_deep(model, verbose=verbose)
    if verbose and patched > 0:
        print(f"Patched {patched} VisionTransformer module(s)")
    
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_optimal_coef(
    results,
    metric="avg_normalized_top1",
    minimize=False,
    control_metric=None,
    control_metric_threshold=0.0,
):
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef


def nonlinear_advantage(nonlinear_acc, linear_acc, num_classes):
    """Computes the normalized non-linear advantage of a finetuned model.

    The nonlinear_advantage is defined as:
        error_rate(linear_model) - error_rate(nonlinear_model) / (1 - 1 / num_classes)
    and takes values between [-1, 1]. A value of 0 indicates that the nonlinear
    model is no better than the linear one. Meanwhile, a value of 1 indicates
    that the nonlinear model is perfect and the linear trivial, and a value of
    -1 indicates the opposite.
    """
    return (nonlinear_acc - linear_acc) / (1.0 - 1.0 / num_classes)

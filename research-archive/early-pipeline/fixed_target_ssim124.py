"""Cache only immutable target terms in the existing SSIM implementation.

The actual SSIM expression is compiled from the supplied original source.
No coefficients, convolutions on the prediction, padding or reductions change.
This optional training interface requires fixed, explicitly named RGB targets.
"""
import ast
import inspect
import textwrap

import torch
import torch.nn.functional as F


def _cached_function(module):
    tree = ast.parse(textwrap.dedent(inspect.getsource(module._ssim)))
    function = tree.body[0]
    names = {'mu2', 'mu2_sq', 'sigma2_sq'}
    if 'valid_mask' in inspect.signature(module._ssim).parameters:
        names.add('support')
    seen = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                seen.append(target.id)
                node.value = ast.parse(f'target_stats[{target.id!r}]', mode='eval').body
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'not bool(support.any())':
            node.test = ast.parse("target_stats['support_empty']", mode='eval').body
    if sorted(seen) != sorted(names):
        raise ValueError('Original SSIM structure changed; inspect before adapting')
    function.args.kwonlyargs.append(ast.arg(arg='target_stats'))
    function.args.kw_defaults.append(None)
    tree = ast.fix_missing_locations(tree)
    namespace = dict(module.__dict__)
    exec(compile(tree, inspect.getsourcefile(module._ssim) + ':fixed-target', 'exec'), namespace)
    return namespace['_ssim'], ast.unparse(tree)


class FixedTargetSSIM:
    def __init__(self, module, targets, frame_ids, validities=None, window_size=11):
        if not targets or len(targets) != len(frame_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError('Require unique frame identities and one target each')
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError('Window size must be positive and odd')
        self.module = module
        self.window_size = window_size
        self.function, self.derived_source = _cached_function(module)
        self.masked_signature = 'valid_mask' in inspect.signature(module._ssim).parameters
        validities = [None] * len(targets) if validities is None else list(validities)
        if len(validities) != len(targets):
            raise ValueError('Validity/frame count mismatch')
        first = targets[0]
        self.window = module.create_window(window_size, first.size(-3)).to(first.device).type_as(first)
        self.records = {}
        cached_tensors = [self.window]
        with torch.no_grad():
            for frame, target, valid in zip(frame_ids, targets, validities):
                if target.requires_grad or target.ndim != 3 or target.shape[0] != 3 or target.dtype != torch.float32:
                    raise ValueError('Expected fixed float32 CHW RGB target without gradients')
                if target.device != first.device:
                    raise ValueError('Target device mismatch')
                if valid is not None and not self.masked_signature:
                    raise ValueError('Original module does not support validity')
                if valid is not None:
                    module._check_validity(target, target, valid)
                    prepared = module._known_only(target, valid)
                    support = module._ssim_support(valid, window_size)
                    if not bool(support.any()):
                        raise ValueError('No complete known SSIM neighborhoods')
                    cached_tensors.extend([prepared, support])
                else:
                    prepared, support = target, None
                mu2 = F.conv2d(prepared, self.window, padding=window_size // 2, groups=3)
                mu2_sq = mu2.pow(2)
                sigma2_sq = F.conv2d(prepared * prepared, self.window, padding=window_size // 2, groups=3) - mu2_sq
                stats = dict(mu2=mu2, mu2_sq=mu2_sq, sigma2_sq=sigma2_sq,
                             support=support, support_empty=False)
                self.records[frame] = dict(target=target, target_version=target._version,
                    valid=valid, valid_version=valid._version if valid is not None else None,
                    prepared=prepared, stats=stats)
                cached_tensors.extend([mu2, mu2_sq, sigma2_sq])
        self.additional_bytes = sum(t.numel() * t.element_size() for t in cached_tensors)

    def ssim(self, prediction, target, *, frame_id):
        if frame_id not in self.records:
            raise ValueError('Unknown SSIM frame identity')
        record = self.records[frame_id]
        if target is not record['target'] or target._version != record['target_version'] or target.requires_grad:
            raise ValueError('Target identity or content changed; rebuild cache explicitly')
        if prediction.shape != target.shape or prediction.dtype != target.dtype or prediction.device != target.device:
            raise ValueError('Prediction/target shape, dtype or device mismatch')
        valid = record['valid']
        if valid is not None:
            if valid._version != record['valid_version']:
                raise ValueError('Validity changed; rebuild cache explicitly')
            prediction = self.module._known_only(prediction, valid)
        kwargs = {'valid_mask': valid} if self.masked_signature else {}
        return self.function(prediction, record['prepared'], self.window, self.window_size, 3, True,
                             target_stats=record['stats'], **kwargs)

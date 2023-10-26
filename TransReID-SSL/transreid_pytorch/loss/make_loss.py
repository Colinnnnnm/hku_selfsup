# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.cuda
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch
import torch.distributed as dist
from torch import Tensor
from typing import Optional, Tuple


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.

    This code was taken and adapted from here:
    https://github.com/Spijkervet/SimCLR

    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input)
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def gather(input: torch.Tensor) -> Tuple[torch.Tensor]:
    """Gathers this tensor from all processes. Supports backprop."""
    return GatherLayer.apply(input)


class VICRegLoss(torch.nn.Module):
    """Implementation of the VICReg loss [0].

    This implementation is based on the code published by the authors [1].

    - [0] VICReg, 2022, https://arxiv.org/abs/2105.04906
    - [1] https://github.com/facebookresearch/vicreg/

    Attributes:
        lambda_param:
            Scaling coefficient for the invariance term of the loss.
        mu_param:
            Scaling coefficient for the variance term of the loss.
        nu_param:
            Scaling coefficient for the covariance term of the loss.
        gather_distributed:
            If True then the cross-correlation matrices from all gpus are gathered and
            summed before the loss calculation.
        eps:
            Epsilon for numerical stability.

    Examples:

        >>> # initialize loss function
        >>> loss_fn = VICRegLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        gather_distributed: bool = False,
        eps=0.0001,
    ):
        super(VICRegLoss, self).__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Returns VICReg loss.

        Args:
            z_a:
                Tensor with shape (batch_size, ..., dim).
            z_b:
                Tensor with shape (batch_size, ..., dim).
        """
        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert (
            z_a.shape == z_b.shape
        ), f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        # invariance term of the loss
        inv_loss = invariance_loss(x=z_a, y=z_b)

        # gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        var_loss = 0.5 * (
            variance_loss(x=z_a, eps=self.eps) + variance_loss(x=z_b, eps=self.eps)
        )
        cov_loss = covariance_loss(x=z_a) + covariance_loss(x=z_b)

        loss = (
            self.lambda_param * inv_loss
            + self.mu_param * var_loss
            + self.nu_param * cov_loss
        )
        return loss


def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
    """Returns VICReg invariance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        y:
            Tensor with shape (batch_size, ..., dim).
    """
    return F.mse_loss(x, y)


def variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    """Returns VICReg variance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        eps:
            Epsilon for numerical stability.
    """
    x = x - x.mean(dim=0)
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))
    return loss


def covariance_loss(x: Tensor) -> Tensor:
    """Returns VICReg covariance loss.

    Generalized version of the covariance loss with support for tensors with more than
    two dimensions. Adapted from VICRegL:
    https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L299

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
    """
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)
    # cov has shape (..., dim, dim)
    cov = torch.einsum("b...c,b...d->...cd", x, x) / (batch_size - 1)
    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=torch.cuda.is_available())  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if cfg.SOLVER.USE_CONTRASTIVE:
        criterion = VICRegLoss()

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target,target_cam, mapped_feat=None):
            return (cfg.MODEL.ID_LOSS_WEIGHT * F.cross_entropy(score, target) +
                    cfg.MODEL.CONTRASTIVE_LOSS_WEIGHT * (criterion(*mapped_feat) if cfg.SOLVER.USE_CONTRASTIVE else 0))

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam, mapped_feat=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    return (cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS +
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS +
                            cfg.MODEL.CONTRASTIVE_LOSS_WEIGHT * (criterion(*mapped_feat) if cfg.SOLVER.USE_CONTRASTIVE else 0))
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    return (cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS +
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS +
                            cfg.MODEL.CONTRASTIVE_LOSS_WEIGHT * (criterion(*mapped_feat) if cfg.SOLVER.USE_CONTRASTIVE else 0))
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion



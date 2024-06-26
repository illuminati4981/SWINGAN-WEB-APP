# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from re import X
import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import dnnlib
# from customed_vgg import Custom_VGG
# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(
        self, phase, real_img, deg_img, real_c, gen_z, gen_c, sync, gain
    ):  # to be overridden by subclass
        raise NotImplementedError()


def matching_loss(gen_feats, real_feats):
    loss = 0
    for (gen_feat, real_feat) in zip(gen_feats, real_feats):
      loss += torch.nn.functional.l1_loss(input=gen_feat, target=real_feat)

    return loss


# ----------------------------------------------------------------------------


class StyleGAN2Loss(Loss):
    def __init__(
        self,
        device,
        swin,
        G_mapping,
        G_synthesis,
        D,
        augment_pipe=None,
        style_mixing_prob=0.9,
        r1_gamma=10,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_weight=2,
    ):
        super().__init__()
        self.device = device
        self.swin = swin
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, x, z, c, sync):
        with misc.ddp_sync(self.swin, sync):
            ### Normalization for swin transformer
            # swin_normalization = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            # swin_input = swin_normalization(x)  

            swin_input = x  
            x, size128_output, size64_output, size32_output, size16_output, size8_output, size4_output = self.swin(swin_input, 16)
            noises = [size128_output, size64_output, size32_output, size16_output, size8_output, size4_output]

        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(x, c)

            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function("style_mixing"):
                    cutoff = torch.empty(
                        [], dtype=torch.int64, device=ws.device
                    ).random_(1, ws.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws.device) < self.style_mixing_prob,
                        cutoff,
                        torch.full_like(cutoff, ws.shape[1]),
                    )
                    ws[:, cutoff:] = self.G_mapping(
                        torch.randn_like(x), c, skip_w_avg_update=True
                    )[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws, noises)

        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits, feats = self.D(img, c)
        return logits, feats

    def accumulate_gradients(self, phase, real_img, deg_img, real_c, gen_z, gen_c, sync, gain):
        ###  TODO: adding deg_img into params and the abstract class function
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth"]
        do_Gmain = phase in ["Gmain", "Gboth"]
        do_Dmain = phase in ["Dmain", "Dboth"]
        do_Gpl = (phase in ["Greg", "Gboth"]) and (self.pl_weight != 0)
        do_Dr1 = (phase in ["Dreg", "Dboth"]) and (self.r1_gamma != 0)
        loss_value = 0


        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function("Gmain_forward"):
                gen_img, _gen_ws = self.run_G(
                    deg_img,
                    gen_z,
                    gen_c,
                    sync=(sync and not do_Gpl),  # deg_img manually added
                )  # May get synced by Gpl.

                gen_logits, gen_feats = self.run_D(gen_img, gen_c, sync=False)
                _, real_feats = self.run_D(real_img, gen_c, sync=False)

                # MSSSIM loss
                # from training.ms_ssim_l1_loss import MS_SSIM_L1_LOSS
                # real_img, gen_img = real_img + 1, gen_img + 1
                # mix_loss = MS_SSIM_L1_LOSS(alpha=0.5, compensation=1)
                # # mix_loss = MS_SSIM_L1_LOSS(alpha=0.5, compensation=200)
                # mix_loss_value = mix_loss(real_img, gen_img)
                # real_img, gen_img = real_img - 1, gen_img - 1
                
                # loss_Gmain = (lpips_value * 0.35 + mix_loss_value / 200 * 0.65) * 200
                softplus = torch.nn.functional.softplus(
                    -gen_logits # -log(sigmoid(gen_logits))
                ).mean()
                # loss_Gmain = (softplus * 0.5 + mix_loss_value * 0.7)
                # loss_Gmain = (softplus / 10 * 0.3 + mix_loss_value * 0.7) * 200
                # loss_Gmain = softplus * 0.3 + torch.nn.functional.l1_loss(real_img, gen_img) * 0.5

                loss_m = matching_loss(real_feats, gen_feats)
                l1_loss = torch.nn.functional.l1_loss(real_img, gen_img)
                loss_Gmain = softplus + loss_m + l1_loss * 0.02

                # id_loss_value = id_loss(gen_img, real_img, self.swin, batch_size=16) * 10000
                # l1_loss = torch.nn.functional.l1_loss(input=real_img, target=gen_img)
                # loss_Gmain = (softplus / 10 * 0.3 + id_loss_value * 0.6 + l1_loss * 0.1) * 200

                # mix loss alpha 0.5 compensation = 200
                
                # Last one
                # loss_Gmain = (softplus / 10 * 0.2 + mix_loss_value / 200 * 0.8) * 200
                # mix loss alpha 0.5 compensation = 200

                # Last two
                # loss_Gmain = (softplus / 10 * 0.2 + mix_loss_value / 200 * 0.8) * 200
                # mix loss alpha 0.2 compensation = 1 u-net 0.5 lr 0.0025

                print('Gmain loss_m: ', loss_m.item())
                print('Gmain l1_loss: ', l1_loss.item())
                # print('Gmain softplus: ', softplus.item())
                # print('Gmain logits: ', gen_logits.mean().item())

            with torch.autograd.profiler.record_function("Gmain_backward"):
                loss_Gmain.mul(gain).backward()
            loss_value = loss_Gmain.mul(gain)


        # Gpl: Apply path length regularization.
        # if do_Gpl:
        #     with torch.autograd.profiler.record_function("Gpl_forward"):
        #          batch_size = deg_img.shape[0] #// self.pl_batch_shrink 

        #          gen_img, gen_ws = self.run_G(
        #              deg_img[:batch_size],  # manually added
        #              gen_z[:batch_size],
        #              gen_c[:batch_size],
        #              sync=sync,
        #          )
        #          pl_noise = torch.randn_like(gen_img) / np.sqrt(
        #              gen_img.shape[2] * gen_img.shape[3]
        #          )
        #          with torch.autograd.profiler.record_function(
        #              "pl_grads"
        #          ), conv2d_gradfix.no_weight_gradients():
        #              pl_grads = torch.autograd.grad(
        #                  outputs=[(gen_img * pl_noise).sum()],
        #                  inputs=[gen_ws],
        #                  create_graph=True,
        #                  only_inputs=True,
        #              )[0]
        #          pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        #          pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        #          self.pl_mean.copy_(pl_mean.detach())
        #          pl_penalty = (pl_lengths - pl_mean).square()
        #          loss_Gpl = pl_penalty * self.pl_weight
        #     with torch.autograd.profiler.record_function("Gpl_backward"):
        #          (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()


        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function("Dgen_forward"):
                gen_img, _gen_ws = self.run_G(
                    deg_img, gen_z, gen_c, sync=False
                )  # real image manually added
                gen_logits, _= self.run_D(
                    gen_img, gen_c, sync=False
                )  # Gets synced by loss_Dreal.

                loss_Dgen = torch.nn.functional.softplus(
                    gen_logits  # -log(1 - sigmoid(gen_logits))  
                ).mean()
                print('Dmain Fake logits: ', gen_logits.mean().item())
            with torch.autograd.profiler.record_function("Dgen_backward"):
                # (loss_Dgen).mul(gain).backward()
                # (loss_Dgen / 10 * 200).mul(gain).backward()
                (loss_Dgen).mul(gain).backward()


        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain:
            name = (
                "Dreal_Dr1" if do_Dmain and do_Dr1 else "Dreal" if do_Dmain else "Dr1"
            )
            with torch.autograd.profiler.record_function(name + "_forward"):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, _= self.run_D(real_img_tmp, real_c, sync=sync)
                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(
                        -real_logits
                    ).mean()  # -log(sigmoid(real_logits))

                print('Dmain Real logits: ', real_logits.mean().item())
                loss_Dr1 = 0
                # if do_Dr1:
                #     with torch.autograd.profiler.record_function(
                #         "r1_grads"
                #     ), conv2d_gradfix.no_weight_gradients():
                #         r1_grads = torch.autograd.grad(
                #             outputs=[real_logits.sum()],
                #             inputs=[real_img_tmp],
                #             create_graph=True,
                #             only_inputs=True,
                #         )[0]
                #     r1_penalty = r1_grads.square().sum([1, 2, 3])
                #     loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                #     training_stats.report("Loss/r1_penalty", r1_penalty)
                #     training_stats.report("Loss/D/reg", loss_Dr1)

            with torch.autograd.profiler.record_function(name + "_backward"):
                # (real_logits * 0 + loss_Dreal / 10 * 200 + loss_Dr1).backward()
                # (loss_Dreal).backward()
                # (loss_Dreal / 10 * 200).backward()
                (loss_Dreal).backward()

        
        return loss_value


# ----------------------------------------------------------------------------
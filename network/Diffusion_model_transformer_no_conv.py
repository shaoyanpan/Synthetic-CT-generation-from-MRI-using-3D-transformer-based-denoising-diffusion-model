from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from log_nnFormer import *
# from SwinUnetr import *
# from .fp16_util import convert_module_to_f16, convert_module_to_f32
from util_nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from monai.utils import ensure_tuple_rep

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, sample_kernel, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if dims == 3:
            self.sample_kernel=(sample_kernel[0],sample_kernel[1],sample_kernel[2])
        else:
            self.sample_kernel=(sample_kernel[0],sample_kernel[1])            
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
        else:
            self.up = th.nn.Upsample(scale_factor=self.sample_kernel,mode='nearest')
            self.conv = conv_nd(dims, self.channels, self.channels, 1, padding=0)
            
    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.up(x)
        x = self.conv(x)
        
        # if self.dims == 3:
        #     x = F.interpolate(
        #         x, scale_factor=self.sample_kernel, mode="nearest"
        #     )
        # else:
        #     x = F.interpolate(x, scale_factor=self.sample_kernel, mode="nearest")
        # if self.use_conv:
        #     x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv,sample_kernel, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if self.dims == 3:
            self.sample_kernel = (1/sample_kernel[0],1/sample_kernel[1],1/sample_kernel[2])
        else:
            self.sample_kernel = (1/sample_kernel[0],1/sample_kernel[1])
        # stride = 2 if dims != 3 else (2, 2, 2)
        # stride = 2
        if use_conv:
            self.op =  th.nn.Upsample(scale_factor=self.sample_kernel,mode='nearest')
        else:
            assert self.channels == self.out_channels
            self.op = th.nn.Upsample(scale_factor=self.sample_kernel,mode='nearest')
            self.conv = conv_nd(dims, self.channels, self.channels, 1, padding=0)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # x = F.interpolate(
        #     x, scale_factor=self.sample_kernel, mode="nearest"
        # )
        return self.conv(self.op(x))


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        sample_kernel = None,
        use_swin = False,
        num_heads = 4,
        window_size = [4,4,4],
        input_resolution = [1,1,1],
        drop_path = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.input_resolution=input_resolution
        self.use_swin=use_swin

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False,sample_kernel, dims)
            self.x_upd = Upsample(channels, False,sample_kernel, dims)
        elif down:
            self.h_upd = Downsample(channels, False,sample_kernel, dims)
            self.x_upd = Downsample(channels, False,sample_kernel, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
            
        if use_swin:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, 1, padding=0),
                    )
            
            self.shift_size = tuple(i // 2 for i in window_size)
            self.no_shift = tuple(0 for i in window_size)
            self.swin_layer = nn.ModuleList([
                                SwinTransformerBlock(
                                    dim=self.out_channels,
                                    input_resolution=self.input_resolution,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=4,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop=0,
                                    attn_drop=0,
                                    drop_path=drop_path,
                                    norm_layer = nn.LayerNorm)
                                for i in range(2)])

            self.out_layers = nn.Sequential(
                    normalization(self.out_channels),
                    nn.Identity())
        else:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
                    )
            self.swin_layer = nn.ModuleList([nn.Identity()])
            self.out_layers = nn.Sequential(
                    normalization(self.out_channels),
                    nn.SiLU(),
                    nn.Dropout(p=0),
                    zero_module(
                        conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                    ),
                )
            
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )


        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            S, H, W = h.size(2), h.size(3), h.size(4)
            h = h.flatten(2).transpose(1, 2).contiguous()
            for blk in self.swin_layer:
                h = blk(h)
            h = h.transpose(1, 2).contiguous().view(-1, self.out_channels, S, H, W)
            # h = self.swin_layer(h)
            h = out_rest(h)
        else:
            h = h + emb_out
            for blk in self.swin_layer:
                h = blk(h)
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class SwinVITModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=False,
        dims=2,
        sample_kernel=None,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        window_size = 4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.sample_kernel = sample_kernel[0]
        spatial_dims = dims
        drop_path = [x.item() for x in th.linspace(0, dropout, len(channel_mult))]
        
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = image_size

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                if ds[0] in attention_resolutions:
                    use_swin = True
                else:
                    use_swin = False
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_swin = use_swin,
                        num_heads = num_heads[level],
                        window_size = window_size[level],
                        input_resolution = ds,
                        drop_path = drop_path[level]
                    )
                ]
                ch = int(mult * model_channels)
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) -1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(mult * model_channels),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_swin = use_swin,
                            num_heads = num_heads[level],
                            window_size = window_size[level],
                            input_resolution = ds,
                            drop_path = drop_path[level],
                            down=True,
                            sample_kernel=self.sample_kernel[level],
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample,self.sample_kernel[level], dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                if dims == 3:
                   ds = [ds[0]//self.sample_kernel[level][0],ds[1]//self.sample_kernel[level][1],ds[2]//self.sample_kernel[level][2]]
                else:
                    ds = [ds[0]//self.sample_kernel[level][0],ds[1]//self.sample_kernel[level][1]]
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=int(mult * model_channels),
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_swin = use_swin,
                num_heads = num_heads[level],
                window_size = window_size[level],
                input_resolution = ds,
                drop_path = drop_path[level]
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=int(mult * model_channels),
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_swin = use_swin,
                num_heads = num_heads[level],
                window_size = window_size[level],
                input_resolution = ds,
                drop_path = drop_path[level]
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                if ds[0] in attention_resolutions:
                    use_swin = True
                else:
                    use_swin = False
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_swin = use_swin,
                        num_heads = num_heads[level],
                        window_size = window_size[level],
                        input_resolution = ds,
                        drop_path = drop_path[level]
                    )
                ]
                ch = int(model_channels * mult)
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads_upsample,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(model_channels * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_swin = use_swin,
                            num_heads = num_heads[level],
                            window_size = window_size[level],
                            input_resolution = ds,
                            drop_path = drop_path[level],
                            up=True,
                            sample_kernel=self.sample_kernel[level-1],
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample,self.sample_kernel[level-1], dims=dims, out_channels=out_ch)
                    )
                    if dims == 3:
                        ds = [ds[0]*self.sample_kernel[level-1][0],
                              ds[1]*self.sample_kernel[level-1][1],
                              ds[2]*self.sample_kernel[level-1][2]]
                    else:
                        ds = [ds[0]*self.sample_kernel[level-1][0],
                              ds[1]*self.sample_kernel[level-1][1]]
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
    
    def forward(self, x, timesteps,cond = None,null_cond_prob = 0., y=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
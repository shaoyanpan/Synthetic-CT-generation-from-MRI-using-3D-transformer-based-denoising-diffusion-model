# Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model
**This is the repository for the paper "[Synthetic CT generation from MRI using 3D transformer-based denoising diffusion model](https://iopscience.iop.org/article/10.1088/1361-6560/acca5c/meta](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16847))".**

The codes were created based on [image-guided diffusion](https://github.com/openai/guided-diffusion), [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet), and [Monai](https://monai.io/)

Updated 1.1:
With a modified variational bound loss code following the image-guided diffusion, we are able to use 1000 training timestep and 50 inference timesteps (instead of 4000 training and 500 inference timesteps in the paper) and stablize the training process to generate the fancy images! **Maybe this is not very important for 2D synthesis, but it is critical for 3D synthesis!!**
The details are shown in our another paper "[Synthetic CT Generation from MRI using 3D Transformer-based Denoising Diffusion Model](https://arxiv.org/abs/2305.19467)"

# Required packages

The requires packages are in test_env.yaml.

Create an environment using Anaconda:
```
conda env create -f \your directory\test_env.yaml
```


# Usage

The usage is in the jupyter notebook TDM main.ipynb. Including how to build a diffusion process, how to build a network, and how to call the diffusion process to train, and sample new synthetic images. However, we give simple example below:

**Create diffusion**
```
from diffusion.Create_diffusion import *
from diffusion.resampler import *

diffusion = create_gaussian_diffusion(
    steps=1000,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule='linear',
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=True,
    rescale_learned_sigmas=True,
    timestep_respacing=[250],
)
schedule_sampler = UniformSampler(diffusion)
```

**Create network**
```
attention_resolutions="64,32,16,8"
attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(int(res))

image_size = 256
from network.Diffusion_model_transformer import *
model = SwinVITModel(
        image_size=(image_size,image_size),
        in_channels=1,
        model_channels=128,
        out_channels=2,
        sample_kernel=([2,2],[2,2],[2,2],[2,2],[2,2]),
        num_res_blocks=[2,2,1,1,1,1],
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_classes=None,
        num_heads=[4,4,4,8,16,16],
        window_size = [[4,4],[4,4],[4,4],[8,8],[8,8],[4,4]],
        use_scale_shift_norm=True,
        resblock_updown=False,
    )
```

**Train the diffusion**
```
batch_size = 10
t, weights = schedule_sampler.sample(batch_size, device)
all_loss = diffusion.training_losses(model,traindata,t=t)
loss = (all_loss["loss"] * weights).mean()
```

**generate new synthetic images**
```
num_sample = 10
image_size = 256
x = diffusion.p_sample_loop(model,(num_sample, 1, image_size, image_size),clip_denoised=True)
```


# Visual examples

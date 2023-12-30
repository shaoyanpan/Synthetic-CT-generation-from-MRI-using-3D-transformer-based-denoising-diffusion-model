
# 2D-Medical-Denoising-Diffusion-Probabilistic-Model
**This is the repository for the paper "[2D Medical Image Synthesis Using Transformer-based Denoising Diffusion Probabilistic Model](https://iopscience.iop.org/article/10.1088/1361-6560/acca5c/meta)".**

The codes were created based on [image-guided diffusion](https://github.com/openai/guided-diffusion), [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet), and [Monai](https://monai.io/)

Notice: Due to the data restriction, we can only provide MATLAB file (so no patient information) with over-smoothed CT volumes. The data we show just to demonstrate how the user should organize their data. The dicom or nii file processing are also included in the Jupyter notebook.

# Required packages

The requires packages are in environment.yaml.

Create an environment using Anaconda:
```
conda env create -f \your directory\environment.yaml
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

![image_1](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/3a814bd3-1107-4d23-b295-9088530754d8)
![image_2](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/cfb2d2c8-f611-497c-93ff-99b7f1ad27a7)
![image_3](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e183a0fd-dcd0-4b1a-8c5f-b861c05b4b9f)
![image_27](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/6c43ef4a-6903-4a72-9363-421fd5c264b4)

![image_4](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/877cfa01-d1b9-4728-ad14-58ac41a3ef9d)
![image_402](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/8c44d75c-7a9b-4de6-ba01-bae18b5dfe2c)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/955b5c65-e4a6-4e08-a870-bd59ad0682bd)
![image_69](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/48f9413e-e630-41e3-9edf-57ad3887822c)

![image_1](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e19f614d-3441-407c-bbbb-e76d2cda6fa3)
![image_5](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/959e8a26-4925-4799-a2b7-a4f8f2e15e43)
![image_7](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/1b4dffb9-a324-4e4b-b76a-1f18648bdb37)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e1300ad7-2a5a-42ea-8980-8f37427ca7b1)

![image_8](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/0ac4a0f3-ce65-4280-8442-ac8f2e000c4d)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/32a0d462-ebbe-465e-9ac2-e8c5d8f75e07)
![image_4](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/f64e4cc0-155d-4b17-b6aa-68d2362be7ec)
![image_46](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/43a3b4ce-7469-4f18-8dd7-87689df410b7)



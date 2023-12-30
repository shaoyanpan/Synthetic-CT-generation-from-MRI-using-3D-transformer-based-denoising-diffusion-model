
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

# Data organization
The data organization example is shown in folder "MRI_to_CT_brain_for_dosimetric\imagesTr". Or you can see the below screenshots:

**MATLAB files: every matlab file can contain a dict has image and label together. So you see you only need two folders: imagesTr for training, imagesTs for testing, and imagesVal for validation. You can change the name but please make sure also change the reading dir in the jupyter notebook**
![Capture](https://github.com/shaoyanpan/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/assets/89927506/1a07d63d-5009-4ecf-aa88-8c86647e46e2)


**Nii files: one nii file can only contain either image or label. So in this case, you need imagesTr and labelsTr for training, imagesTs and labelsTs for testing, and imagesVal and labelsVal for validation**
![Capture2](https://github.com/shaoyanpan/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/assets/89927506/b6f7757c-f962-44ca-974f-266429b6e6f9)


# Usage

The usage is in the jupyter notebook MC-IDDPM main.ipynb. Including how to build a diffusion process, how to build a network, and how to call the diffusion process to train, and sample new synthetic images. However, we give simple example below:

**Create diffusion**
```
from diffusion.Create_diffusion import *
from diffusion.resampler import *

diffusion_steps=1000
learn_sigma=True
timestep_respacing=[50]

# Don't toch these parameters, they are irrelant to the image synthesis
sigma_small=False
class_cond=False
noise_schedule='linear'
use_kl=False
predict_xstart=False
rescale_timesteps=True
rescale_learned_sigmas=True
use_checkpoint=False

diffusion = create_gaussian_diffusion(
    steps=diffusion_steps,
    learn_sigma=learn_sigma,
    sigma_small=sigma_small,
    noise_schedule=noise_schedule,
    use_kl=use_kl,
    predict_xstart=predict_xstart,
    rescale_timesteps=rescale_timesteps,
    rescale_learned_sigmas=rescale_learned_sigmas,
    timestep_respacing=timestep_respacing,
)
schedule_sampler = UniformSampler(diffusion)
```

**Create network**
```
num_channels=64
attention_resolutions="32,16,8"
channel_mult = (1, 2, 3, 4)
num_heads=[4,4,8,16]
window_size = [[4,4,4],[4,4,4],[4,4,2],[4,4,2]]
num_res_blocks = [2,2,2,2]
sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1]),

attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(int(res))
class_cond = False
use_scale_shift_norm=True
resblock_updown = False
dropout = 0

from network.Diffusion_model_transformer import *
model = SwinVITModel(
          image_size=img_size,
          in_channels=2,
          model_channels=num_channels,
          out_channels=2,
          dims=3,
          sample_kernel = sample_kernel,
          num_res_blocks=num_res_blocks,
          attention_resolutions=tuple(attention_ds),
          dropout=dropout,
          channel_mult=channel_mult,
          num_classes=None,
          use_checkpoint=False,
          use_fp16=False,
          num_heads=num_heads,
          window_size = window_size,
          num_head_channels=64,
          num_heads_upsample=-1,
          use_scale_shift_norm=use_scale_shift_norm,
          resblock_updown=resblock_updown,
          use_new_attention_order=False,
      ).to(device)
```

**Train the diffusion**
```
batch_size = 10
t, weights = schedule_sampler.sample(batch_size, device)
all_loss = diffusion.training_losses(model,target,condition, t)
loss = (all_loss["loss"] * weights).mean()
```

**Testing using MONAI's window-sliding inferencer**
```
img_num = 12
overlap = 0.5
inferer = SlidingWindowInferer(img_size, img_num, overlap=overlap, mode ='constant')
def diffusion_sampling(condition, model):
    sampled_images = diffusion.p_sample_loop(model,(condition.shape[0], 1,
                                                    condition.shape[2], condition.shape[3],condition.shape[4]),
                                                    condition = condition,clip_denoised=True)
    return sampled_images

sampled_images = inferer(condition,diffusion_sampling,model)
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



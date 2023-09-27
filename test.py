# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util, logger

from guided_diffusion.clip_guidance import CLIP_gd
from guided_diffusion.guidance import image_loss, text_loss
import clip

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

from guided_diffusion.image_datasets import load_data
from torchvision import utils
import math

# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf):

    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))
    # device = 'cpu'
    
    conf.text_instruction_file = 'ffhq_instructions.txt'
    conf.text_weight = 160
    conf.image_weight = 100
    conf.image_loss = 'semantic'
    conf.clip_path = './data/pretrained/clip_horse.pt'

    # 创建预测噪声模型和扩散模型——默认参数
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # 显示进度条
    show_progress = conf.show_progress
    
    # 创建Clip模型：用Resnet 50x16
    clip_model, preprocess = clip.load('RN50x16', device='cuda')
    if conf.text_weight == 0:
        instructions = [""]
    else:
        with open(conf.text_instruction_file, 'r') as f:
            instructions = f.readlines()
    instructions = [tmp.replace('\n', '') for tmp in instructions]
    
    # 微调后的clip模型(基于噪声图像输入的微调clip模型)
    clip_ft = CLIP_gd(conf)
    clip_ft.load_state_dict(th.load(conf.clip_path, map_location='cpu'))
    clip_ft = clip_ft.to(device)
    clip_ft.eval()
    
    # 引导函数F(x_t, x'_t, t)对x_t的梯度
    def cond_fn_sdg(x, t, y, **kwargs):
        assert y is not None
        # print("cond_fn_sdg")
        with th.no_grad():
            text_features = clip_model.encode_text(y)
            target_img_noised = diffusion.q_sample(kwargs['ref_img'], t)
            target_img_features = clip_ft.encode_image_list(target_img_noised, t)
            # print(len(target_img_features))
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            image_features = clip_ft.encode_image_list(x_in, t)
            if conf.text_weight != 0:
                loss_text = text_loss(image_features, text_features, conf)
            else:
                loss_text = 0
            if conf.image_weight != 0:
                loss_img = image_loss(image_features, target_img_features, conf)
            else:
                loss_img = 0
            # print(loss_text)
            total_guidance = loss_img * conf.image_weight + loss_text * conf.text_weight
            return th.autograd.grad(total_guidance.sum(), x_in)[0]

    ref = load_reference(
        "ref_imgs",
        1,
        image_size=256,
        class_cond=False,
    )
    
    #定义引导函数cond_fn_sdg
    if conf.image_weight == 0:
        cond_fn = None
    else:
        cond_fn = cond_fn_sdg
    
    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []
    
    # 导入ref
    model_kwargs1 = next(ref)
#     model_kwargs1 = {k: v.to(dist_util.dev()) for k, v in model_kwargs1.items()}
    model_kwargs1 = {k: v.to("cuda") for k, v in model_kwargs1.items()}

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["ref_img"] = model_kwargs1["ref_img"]

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        # if conf.cond_y is not None:
        #     classes = th.ones(batch_size, dtype=th.long, device=device)
        #     model_kwargs["y"] = classes * conf.cond_y
        # else:
        #     classes = th.randint(
        #         low=0, high=NUM_CLASSES, size=(batch_size,), device=device
        #     )
        #     model_kwargs["y"] = classes
        instruction = instructions[0] 
        text = clip.tokenize([instruction for cnt in range(batch_size)]).to('cuda')
        model_kwargs['y'] = text

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            range_t=0,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)

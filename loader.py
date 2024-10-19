# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from iBOT library:
https://github.com/bytedance/ibot
"""

import random
import math
import numpy as np
import os,json
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class Reflacx(Dataset):
    def __init__(self, data_root: str, transforms=None) -> None:
        
        self.data_root=data_root
        self.info=json.load(open(os.path.join(self.data_root,"reflacx.json")))
        self.gaze_dir=os.path.join(self.data_root,"attention")
        self.transforms=transforms
        # self.vis_trans=PairedTransform(transforms[0])
        # self.val_trans=transforms[1]

    def getImgPath(self,index):
        image_path=self.info[index]['image_path']
        image_path=os.path.join(self.data_root,image_path)
        return image_path

    def __getitem__(self, index):
        image_path=self.info[index]['image_path']
        study_id=self.info[index]['study_id']
        reflacx_id=self.info[index]['reflacx_id']
        image_path=os.path.join(self.data_root,image_path)
        # gaze_path=os.path.join(self.gaze_dir,study_id,f"{reflacx_id}.png")

        image = Image.open(image_path).convert('RGB')
        # gaze=Image.open(gaze_path)
        
        if self.transforms !=None:
            # image,gaze=self.transforms(image,gaze)
            image=self.transforms(image)
        return image
        # return {"image":image}
        # return {"image":image,"gaze":gaze}
    

    def __len__(self):
        return len(self.info)



class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class ImageFolderMask(Reflacx):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
                masks.append(mask)
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
                masks.append(mask)
            elif self.pred_shape in ['attmask_high', 'attmask_hint', 'attmask_low']:
                pass
                # masks.append(None)
            
            else:
                # no implementation
                assert False

        # for img in output[0]:
        #     try:
        #         H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
        #     except:
        #         # skip non-image
        #         continue
            
        #     high = self.get_pred_ratio() * H * W
            
        #     if self.pred_shape == 'block':
        #         # following BEiT (https://arxiv.org/abs/2106.08254), see at
        #         # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
        #         mask = np.zeros((H, W), dtype=bool)
        #         mask_count = 0
        #         while mask_count < high:
        #             max_mask_patches = high - mask_count

        #             delta = 0
        #             for attempt in range(10):
        #                 low = (min(H, W) // 3) ** 2 
        #                 target_area = random.uniform(low, max_mask_patches)
        #                 aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
        #                 h = int(round(math.sqrt(target_area * aspect_ratio)))
        #                 w = int(round(math.sqrt(target_area / aspect_ratio)))
        #                 if w < W and h < H:
        #                     top = random.randint(0, H - h)
        #                     left = random.randint(0, W - w)

        #                     num_masked = mask[top: top + h, left: left + w].sum()
        #                     if 0 < h * w - num_masked <= max_mask_patches:
        #                         for i in range(top, top + h):
        #                             for j in range(left, left + w):
        #                                 if mask[i, j] == 0:
        #                                     mask[i, j] = 1
        #                                     delta += 1

        #                 if delta > 0:
        #                     break

        #             if delta == 0:
        #                 break
        #             else:
        #                 mask_count += delta
        #         masks.append(mask)
        #     elif self.pred_shape == 'rand':
        #         mask = np.hstack([
        #             np.zeros(H * W - int(high)),
        #             np.ones(int(high)),
        #         ]).astype(bool)
        #         np.random.shuffle(mask)
        #         mask = mask.reshape(H, W)
        #         masks.append(mask)
        #     elif self.pred_shape in ['attmask_high', 'attmask_hint', 'attmask_low']:
        #         pass
            
        #     else:
        #         # no implementation
        #         assert False

        # return output + (masks,)
        return output+[masks,]
        # return output + [masks]
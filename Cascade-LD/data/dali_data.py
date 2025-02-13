import torch
import numpy as np
import random
import json
import os

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

class LaneExternalIterator(object):
    def __init__(self, path, list_path, batch_size=None, mode = 'train', dataset_name=None):
        assert mode in ['train', 'test']
        self.mode = mode
        self.path = path
        self.list_path = list_path
        self.batch_size = batch_size

        if isinstance(list_path, str):
            with open(list_path, 'r') as f:
                total_list = f.readlines()
        elif isinstance(list_path, list) or isinstance(list_path, tuple):
            total_list = []
            for lst_path in list_path:
                with open(lst_path, 'r') as f:
                    total_list.extend(f.readlines())
        else:
            raise NotImplementedError
        if self.mode == 'train':
            if dataset_name == 'CULane':
                cache_path = os.path.join(path, 'culane_anno_cache.json')
            elif dataset_name == 'Tusimple':
                cache_path = os.path.join(path, 'tusimple_anno_cache.json')

            print('loading cached data')
            cache_fp = open(cache_path, 'r')
            self.cached_points = json.load(cache_fp)
            print('cached data loaded')

        self.total_len = len(total_list)
    
        self.list = total_list
        self.n = len(self.list)

    def __iter__(self):
        self.i = 0
        if self.mode == 'train':
            random.shuffle(self.list)
        return self

    def _prepare_train_batch(self):
        images = []
        seg_images = []

        for _ in range(self.batch_size):
            l = self.list[self.i % self.n]
            l_info = l.split()
            img_name = l_info[0]
            seg_name = l_info[1]

            if img_name[0] == '/':
                img_name = img_name[1:]
            if seg_name[0] == '/':
                seg_name = seg_name[1:]
                
            img_name = img_name.strip()
            seg_name = seg_name.strip()
            
            img_path = os.path.join(self.path, img_name)
            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))

            img_path = os.path.join(self.path, seg_name)
            with open(img_path, 'rb') as f:
                seg_images.append(np.frombuffer(f.read(), dtype=np.uint8))

            self.i = self.i + 1
            
        return (images, seg_images)

    
    def _prepare_test_batch(self):
        images = []
        names = []
        for _ in range(self.batch_size):
            img_name = self.list[self.i % self.n].split()[0]

            if img_name[0] == '/':
                img_name = img_name[1:]
            img_name = img_name.strip()

            img_path = os.path.join(self.path, img_name)

            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))
            names.append(np.array(list(map(ord,img_name))))
            self.i = self.i + 1
            
        return images, names

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        if self.mode == 'train':
            res = self._prepare_train_batch()
        elif self.mode == 'test':
            res = self._prepare_test_batch()
        else:
            raise NotImplementedError

        return res
    def __len__(self):
        return self.total_len

    next = __next__

def encoded_images_sizes(jpegs):
    shapes = fn.peek_image_shape(jpegs)  # the shapes are HWC
    h = fn.slice(shapes, 0, 1, axes=[0]) # extract height...
    w = fn.slice(shapes, 1, 1, axes=[0]) # ...and width...
    return fn.cat(w, h)               # ...and concatenate

def ExternalSourceTrainPipeline(batch_size, num_threads, external_data, train_width, train_height, num_lanes):
    # check cuda
    if torch.cuda.is_available():
        pipe = Pipeline(batch_size, num_threads, device_id=0)
    else:
        pipe = Pipeline(batch_size, num_threads)
    with pipe:
        jpegs, seg_images, labels = fn.external_source(source=external_data, num_outputs=3)
        images = fn.decoders.image(jpegs, device="mixed")
        seg_images = fn.decoders.image(seg_images, device="mixed", output_type=types.GRAY)
        size = encoded_images_sizes(jpegs)
        center = size / 2

        # Resize the images
        images = fn.resize(images, resize_x=train_width, resize_y=train_height)
        seg_images = fn.resize(seg_images, resize_x=train_width, resize_y=train_height, interp_type=types.INTERP_NN)
        # reshape the seg_images to HxW
        seg_images = fn.reshape(seg_images, shape=[train_height, train_width])
        # One hot encoding
        seg_images = fn.one_hot(seg_images, num_classes=num_lanes+1)

        # Data augmentation
        # Random scaling
        mt = fn.transforms.scale(scale = fn.random.uniform(range=(0.8, 1.2), shape=[2]), center = center)
        # Random rotation -6 to 6 degrees
        mt = fn.transforms.rotation(mt, angle = fn.random.uniform(range=(-6, 6)), center = center)
        # Random value -200 to 200 pixels in x and -100 to 100 pixels in y
        off = fn.cat(fn.random.uniform(range=(-200, 200), shape = [1]), fn.random.uniform(range=(-100, 100), shape = [1]))
        # Apply the translation
        mt = fn.transforms.translation(mt, offset = off)

        # Apply the transformation and fill the area that is not covered by black
        images = fn.warp_affine(images, matrix = mt, fill_value=0, inverse_map=False)
        seg_images = fn.warp_affine(seg_images, matrix = mt, fill_value=0, inverse_map=False)
        

        # Normalize the images
        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            # mean of data
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            # std of data
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        seg_images = fn.transpose(seg_images, perm=[2, 0, 1])

        pipe.set_outputs(images, seg_images)
    return pipe

def ExternalSourceTestPipeline(batch_size, num_threads, external_data, train_width, train_height):
    pipe = Pipeline(batch_size, num_threads, device_id=0)
    with pipe:
        jpegs, names = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")

        images = fn.resize(images, resize_x=train_width, resize_y=train_height)
        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255])

        names = fn.pad(names, axes=0, fill_value = -1, shape = 46)
        pipe.set_outputs(images, names)
    return pipe

class TrainCollect:
    def __init__(self, batch_size, num_threads, data_root, list_path, train_width, train_height, num_lanes, dataset_name):
        eii = LaneExternalIterator(data_root, list_path, batch_size=batch_size, dataset_name = dataset_name)
        pipe = ExternalSourceTrainPipeline(batch_size, num_threads, eii, train_width, train_height, num_lanes)
        self.pii = DALIGenericIterator(pipe, output_map = ['images', 'seg_labels'], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        # print('show images and seg_images')
        # # show images and seg_images [0]
        # for data in self.pii:
        #     images = data[0]['images'].cpu().numpy()
        #     seg_images = data[0]['seg_images'].cpu().numpy()
        #     # normalize to 0-255
        #     image = images[0]
        #     image = image.transpose(1,2,0)
        #     image = (image * [0.229 * 255, 0.224 * 255, 0.225 * 255] + [0.485 * 255, 0.456 * 255, 0.406 * 255]).astype(np.uint8)
        #     seg_image = seg_images[0]
        #     print(seg_image.shape)
        #     seg_image = np.argmax(seg_image, axis=0)
        #     seg_image = seg_image.reshape(train_height, train_width)
        #     # plot color lane in image
        #     from functions import color_lanes, blend
        #     out_segmentation_viz = np.zeros((train_height, train_width, 3))

        #     for i in range(1, 5):
        #         rand_c1 = random.randint(1, 255)
        #         rand_c2 = random.randint(1, 255)    
        #         rand_c3 = random.randint(1, 255)
        #         out_segmentation_viz = color_lanes(
        #             out_segmentation_viz, seg_image, 
        #             i, (rand_c1, rand_c2, rand_c3), train_height, train_width)
        #     from PIL import Image
        #     image_orig = Image.fromarray(image)
        #     im_seg = blend(image_orig, out_segmentation_viz)
        #     import matplotlib.pyplot as plt
        #     plt.imshow(im_seg)
        #     plt.show()
        #     break

        self.eii_n = eii.n
        self.batch_size = batch_size
    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.pii)
        images = data[0]['images']
        seg_labels = data[0]['seg_labels']

        return {'images':images, 'seg_labels':seg_labels}
    
    def __len__(self):
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)
    def reset(self):
        self.pii.reset()
    next = __next__


class TestCollect:
    def __init__(self, batch_size, num_threads, data_root, list_path, train_width, train_height):
        self.batch_size = batch_size
        eii = LaneExternalIterator(data_root, list_path, batch_size=batch_size, mode = 'test')
        pipe = ExternalSourceTestPipeline(batch_size, num_threads, eii, train_width, train_height)
        self.pii = DALIGenericIterator(pipe, output_map = ['images', 'names'], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        self.eii_n = eii.n
    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.pii)
        images = data[0]['images']
        names = data[0]['names']
        restored_names = []
        for name in names:
            if name[-1] == -1:
                restored_name = ''.join(list(map(chr,name[:-1])))
            else:
                restored_name = ''.join(list(map(chr,name)))
            restored_names.append(restored_name)
            
        out_dict = {'images': images, 'names': restored_names}
        return out_dict
    
    def __len__(self):
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)

    def reset(self):
        self.pii.reset()
    next = __next__
    
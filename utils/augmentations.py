import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    def __init__(self, mean=(0.485,0.456,0.406), var=(0.229,0.224,0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.var = np.array(var, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.var
        return image, boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        if boxes is not None:
            height, width, channels = image.shape

            wh = np.array([width, height], dtype=np.float32)
            boxes = boxes * np.tile(wh, 4)

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        if boxes is not None:
            height, width, channels = image.shape

            wh = np.array([width, height], dtype=np.float32)
            boxes = boxes / np.tile(wh, 4)
            
        return image, boxes, labels
    
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        h, w, _ = image.shape
        target_h, target_w = self.size
        
        if boxes is not None:
            new_wh = np.array([target_w / w, target_h / h], dtype=np.float32)
            
            boxes = boxes * np.tile(new_wh, 4)

        image = cv2.resize(image.astype(np.uint8), (target_w, target_h))

        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
            image[image>255] = 255
            image[image<0] = 0
        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=36.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image[image>255] = 255
            image[image<0] = 0
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=16):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image[image>255] = 255
            image[image<0] = 0
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            if boxes is not None:
                boxes = boxes.copy() # Nx8
                boxes[:, 0::2] = width - boxes[:, 0::2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class Rotate(object):
    def __init__(self, mean):
        self.angle_option = [0, -90, 180, 90]
        self.deg = 90
        self.rotate_prob = 3     # 5 -> 0.2
        self.mean = mean

    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape
        center = (width/2, height/2)
        size = (width, height)

        if True: #random.randint(self.rotate_prob) == 0:
            angle = random.choice(self.angle_option)
            #angle = random.randint(-self.deg, self.deg)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            image = cv2.warpAffine(image, M, size, borderValue=self.mean, flags=cv2.INTER_LINEAR)
            if boxes is not None:
                for k, box in enumerate(boxes):
                    for l in range(4):
                        pt = np.append(box[2*l:2*(l+1)], 1)
                        rot_pt = M.dot(pt)
                        boxes[k,2*l:2*(l+1)] = rot_pt[:2]
            
        return image, boxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            # RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes=None, labels=None):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        # im, boxes, labels = self.rand_light_noise(im, boxes, labels)
        
        im = np.clip(im, 0., 255.)
        return im, boxes, labels


class Augmentation(object):
    def __init__(self, size, mean=(104, 117, 123), var=(0.229,0.224,0.225)):
        self.mean = mean
        self.size = size

        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            Rotate(self.mean),
            PhotometricDistort(),
            RandomMirror(),
            Resize(self.size), 
            ToPercentCoords(),
            Normalize(mean, var),
        ])

    def __call__(self, img, boxes, labels):
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32)
            
        if labels is not None:
            labels = np.array(labels, dtype=np.int32)
        
        return self.augment(img, boxes, labels)

    
class Augmentation_test(object):
    def __init__(self, size, mean=(104, 117, 123), var=(0.229,0.224,0.225)):
        self.mean = mean
        self.size = size

        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            Resize(self.size), 
            ToPercentCoords(),
            Normalize(mean, var),
        ])

    def __call__(self, img, boxes=None, labels=None):
        
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32)
            
        if labels is not None:
            labels = np.array(labels, dtype=np.int32)
        
        return self.augment(img, boxes, labels)
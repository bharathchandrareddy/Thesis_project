
import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2
import numpy as np


#### Augmentations
def shift_image(img, shift_pnt):
    M = np.float32([[1, 0, shift_pnt[0]], [0, 1, shift_pnt[1]]])
    res = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    return res


def rotate_image(image, angle, scale, rot_pnt):
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) #INTER_NEAREST
    return result


def gauss_noise(img, var=30):
    row, col, ch = img.shape
    mean = var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return np.clip(img.astype(np.int32) + gauss, 0, 255).astype('uint8')


def clahe(img, clipLimit=2.0, tileGridSize=(5,5)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
    return img_output


def _blend(img1, img2, alpha):
    return np.clip(img1 * alpha + (1 - alpha) * img2, 0, 255).astype('uint8')


_alpha = np.asarray([0.114, 0.587, 0.299]).reshape((1, 1, 3))
def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)


def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)


def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)


def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 3)
    return _blend(img, gs, alpha)


def change_hsv(img, h, s, v):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(int)
    hsv[:,:,0] += h
    hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 255)
    hsv[:,:,1] += s
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    hsv[:,:,2] += v
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    hsv = hsv.astype('uint8')
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def shift_channels(img, b_shift, g_shift, r_shift):
    img = img.astype(int)
    img[:,:,0] += b_shift
    img[:,:,0] = np.clip(img[:,:,0], 0, 255)
    img[:,:,1] += g_shift
    img[:,:,1] = np.clip(img[:,:,1], 0, 255)
    img[:,:,2] += r_shift
    img[:,:,2] = np.clip(img[:,:,2], 0, 255)
    img = img.astype('uint8')
    return img
    
def invert(img):
    return 255 - img

def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img
    
#######


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def iou(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    union = np.logical_or(im1, im2)
    im_sum = union.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / im_sum


class ResizeCrop(torch.nn.Module):
    def __init__(self,
                 p: float,
                 input_size: int,
                 weights: Sequence[float] = [
                     1, 9.04788032, 8.68207691, 12.9632271]) -> None:
        """ResizeCrop class.

        Args:
            p: Probability of applying a random size of crop 
                (different from the input size).
            input_size: Size of the input image after cropping and resizing.
            weights: Weights to be applied to the different classes in the
                cropped masks. We have one weight per damage class (4). 
                The default values are the inverse of the frequency of each
                class in the training/validation set with no contamination
                (see generalization section in the paper)
        """

        super().__init__()
        self.p = p
        self.input_size = input_size
        self.w = weights

    def forward(self,
                tensor: torch.Tensor) -> torch.Tensor:

        crop_size = self.input_size
        if random.random() > self.p:
            crop_size = random.randint(
                int(self.input_size / 1.15), int(self.input_size / 0.85))

            # We need to check that the tensor has the expected shape (11, H, W)
            assert tensor.shape[0] == 11

            # Separate img, msk, and aug_img from the concatenated tensor
            img = tensor[:6, ...]      # First 6 channels are img
            msk = tensor[6:, ...]    # Next 5 channels are msk

            bst_x0 = random.randint(0, tensor.shape[1] - crop_size)
            bst_y0 = random.randint(0, tensor.shape[2] - crop_size)
            bst_sc = -1
            try_cnt = random.randint(1, 10)
            for _ in range(try_cnt):
                x0 = random.randint(0, tensor.shape[1] - crop_size)
                y0 = random.randint(0, tensor.shape[2] - crop_size)
                # We try to get more of certain classes in the cropped masks.
                _sc = msk[2, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[1] + \
                    msk[3, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[2] + \
                    msk[4, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[3] + \
                    msk[1, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[0]
                if _sc > bst_sc:
                    bst_sc = _sc
                    bst_x0 = x0
                    bst_y0 = y0
            x0 = bst_x0
            y0 = bst_y0

            # Apply the crop to the entire tensor (img, msk, aug_img)
            tensor = tensor[:, y0:y0 + crop_size, x0:x0 + crop_size]

        # Resize the cropped tensor back to the input size
        tensor = TF.resize(img=tensor,
                        size=[self.input_size, self.input_size],
                        interpolation=transforms.InterpolationMode.NEAREST,
                        antialias=True)
        #print(f'shape of resize crop output={tensor.shape}')
        return tensor
    

class FusionAugmentation:
    def __init__(self):
        pass

    def contrast_enhancement(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, strength=1.5):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(strength + 1) * image - float(strength) * blurred
        sharpened = np.maximum(sharpened, 0)
        sharpened = np.minimum(sharpened, 255)
        sharpened = sharpened.round().astype(np.uint8)
        return sharpened

    def sobel_edge_detection(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # x direction
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # y direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = np.uint8(magnitude)
        return magnitude

    def canny_edge_detection(self,image, lower_threshold=50, upper_threshold=150):

        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

            # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

            # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)
        return edges



    def image_augmentation_rgb(self, image):
        channels = cv2.split(image)
        augmented_channels = []
        for channel in channels:
            enhanced_channel = self.contrast_enhancement(channel)
            unsharp_image = self.unsharp_mask(enhanced_channel)
            edge_image = self.canny_edge_detection(unsharp_image)
            augmented_channels.append(edge_image)
        augmented_image = cv2.merge(augmented_channels)
        return augmented_image

    def __call__(self, image):
        augmented_image = self.image_augmentation_rgb(image)
        tensor_image = torch.tensor(augmented_image, dtype=torch.float32)
        tensor_image = tensor_image / 255.0
        tensor_image = tensor_image.permute(2, 0, 1)
        #print(f'tensor shape of image after augmentation= {tensor_image.shape}')
        return tensor_image
    
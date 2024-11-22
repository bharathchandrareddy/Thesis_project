from typing import Tuple

import albumentations as A
import cv2
import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2
import numpy as np


__all__ = [
    "safe_color_augmentations",
    "safe_spatial_augmentations",
    "light_color_augmentations",
    "light_spatial_augmentations",
    "light_post_image_transform",
    "medium_color_augmentations",
    "medium_spatial_augmentations",
    "medium_post_transform_augs",
    "hard_spatial_augmentations",
    "hard_color_augmentations",
    "old_light_augmentations",
    "old_post_transform_augs"
]


def safe_color_augmentations():
    return A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True)])


def safe_spatial_augmentations(image_size: Tuple[int, int]):
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.MaskDropout(10),
            A.Compose([A.Transpose(), A.RandomRotate90()]),
        ]
    )


def light_color_augmentations():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True),
            A.RandomGamma(gamma_limit=(90, 110)),
        ]
    )


def light_spatial_augmentations(image_size: Tuple[int, int]):
    return A.Compose(
        [
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            # Spatial-preserving augmentations:
            A.RandomBrightnessContrast(),

            A.MaskDropout(max_objects=10),
        ]
    )


def old_light_augmentations(image_size: Tuple[int, int]):
    return A.Compose(
        [
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            # Spatial-preserving augmentations:
            A.RandomBrightnessContrast(),
            A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        ]
    )


def light_post_image_transform():
    return A.OneOf(
        [
            A.NoOp(),
            A.Compose(
                [
                    A.PadIfNeeded(1024 + 10, 1024 + 10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomSizedCrop((1024 - 5, 1024 + 5), 1024, 1024),
                ],
                p=0.2,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                rotate_limit=3,
                scale_limit=0.02,
                border_mode=cv2.BORDER_CONSTANT,
                mask_value=0,
                value=0,
                p=0.2,
            ),
        ]
    )


def old_post_transform_augs():
    return A.OneOf(
        [
            A.NoOp(),
            A.Compose(
                [
                    A.PadIfNeeded(1024 + 20, 1024 + 20, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.RandomSizedCrop((1024 - 10, 1024 + 10), 1024, 1024),
                ],
                p=0.2,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625, rotate_limit=3, scale_limit=0.05, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2
            ),
        ]
    )


def medium_post_transform_augs():
    return A.OneOf(
        [
            A.NoOp(),
            A.Compose(
                [
                    A.PadIfNeeded(1024 + 40, 1024 + 40, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.RandomSizedCrop((1024 - 20, 1024 + 20), 1024, 1024),
                ]
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, rotate_limit=5, scale_limit=0.075, border_mode=cv2.BORDER_CONSTANT, value=0
            ),
        ]
    )


def medium_color_augmentations():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True),
            A.RandomGamma(gamma_limit=(90, 110)),
            A.OneOf(
                [
                    A.NoOp(p=0.8),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                ],
                p=0.2,
            ),
        ]
    )


def medium_spatial_augmentations(image_size: Tuple[int, int], no_mask_dropout=False):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.NoOp(p=0.8),
                    A.RandomGridShuffle(grid=(4, 4), p=0.2),
                    A.RandomGridShuffle(grid=(3, 3), p=0.2),
                    A.RandomGridShuffle(grid=(2, 2), p=0.2),
                ], p=1
            ),

            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            # Spatial-preserving augmentations:
            A.RandomBrightnessContrast(),

            A.NoOp() if no_mask_dropout else A.MaskDropout(max_objects=10),
        ]
    )



def hard_color_augmentations():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True),
            A.RandomGamma(gamma_limit=(90, 110)),
            A.OneOf([A.NoOp(), A.MultiplicativeNoise(), A.GaussNoise(), A.ISONoise()]),
            A.OneOf([A.RGBShift(), A.HueSaturationValue(), A.NoOp()]),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.3),
        ]
    )


def hard_spatial_augmentations(image_size: Tuple[int, int], rot_angle=45):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.NoOp(),
                    A.RandomGridShuffle(grid=(4, 4)),
                    A.RandomGridShuffle(grid=(3, 3)),
                    A.RandomGridShuffle(grid=(2, 2)),
                ]
            ),
            A.MaskDropout(max_objects=10),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        scale_limit=0.1, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                    ),
                    A.NoOp(),
                ]
            ),
            A.OneOf(
                [
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.NoOp(),
                ]
            ),
            # D4
            A.Compose([A.Transpose(), A.RandomRotate90()]),
        ]
    )


def contrast_enhancement( image):
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
    if image is None:
        raise ValueError("Image not found or unable to load.")

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
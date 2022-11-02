import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint
import cv2


def add_elastic_transform(image, mask, alpha, sigma, pad_size=0, seed=None, img_size = 512):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor
        sigma :  σ is an elasticity coefficient
        random_state = random integer
        Return :
        image : elastically transformed numpy array of image
    """
    mask = np.expand_dims(mask, axis=2)
    mask = mask.repeat(3, 2)
    
    image = cv2.resize(image, (img_size, img_size))
    mask = cv2.resize(mask, (img_size, img_size))
    
    image_size = int(image.shape[0])
    mask_size = int(mask.shape[0])
    if seed is None:
        seed = randint(1, 10000)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
    image_shape = image.shape
    mask_shape = mask.shape
    dx = gaussian_filter((random_state.rand(*image_shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*image_shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    
    x, y,z = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]), np.arange(image_shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    res_img = cropping(map_coordinates(image, indices, order=1).reshape(image_shape), img_size, pad_size, pad_size)
    res_mask = cropping(map_coordinates(mask, indices, order=1).reshape(mask_shape), img_size, pad_size, pad_size)
    res_mask = res_mask[:,:,0]
    return res_img,res_mask

def add_gaussian_noise(image, mean=0, std=10.0):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    noise_img = ceil_floor_image(noise_img)
    return noise_img

def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    uni_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    noise_img = ceil_floor_image(noise_img)
    return noise_img

def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

def approximate_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 only with 255 and 0
    """
    image[image > 127.5] = 255
    image[image < 127.5] = 0
    image = image.astype("uint8")
    return image


def normalization1(image, mean, std):
    """ Normalization using mean and std
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """

    image = image / 255.0  # values will lie between 0 and 1.
    m = image.mean()
    s = image.std()
    image = (image - m + mean) / (s/std)

    return image

def normalization2(image, max, min):
    """Normalization to range of [min, max]
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """
    image_new = (image - np.min(image))*(max - min)/(np.max(image)-np.min(image)) + min
    return image_new


def cropping(image, crop_size, dim1, dim2):
    """crop the image and pad it to in_size
    Args :
        images : numpy array of images
        crop_size(int) : size of cropped image
        dim1(int) : vertical location of crop
        dim2(int) : horizontal location of crop
    Return :
        cropped_img: numpy array of cropped image
    """
    cropped_img = image[dim1:dim1+crop_size, dim2:dim2+crop_size]
    return cropped_img

def add_padding(image, in_size, out_size, mode):
    """Pad the image to in_size
    Args :
        images : numpy array of images
        in_size(int) : the input_size of model
        out_size(int) : the output_size of model
        mode(str) : mode of padding
    Return :
        padded_img: numpy array of padded image
    """
    pad_size = int((in_size - out_size)/2)
    padded_img = np.pad(image, pad_size, mode=mode)
    return padded_img


def division_array(crop_size, crop_num1, crop_num2, dim1, dim2):
    """Make division array
    Args :
        crop_size(int) : size of cropped image
        crop_num2 (int) : number of crop in horizontal way
        crop_num1 (int) : number of crop in vertical way
        dim1(int) : vertical size of output
        dim2(int) : horizontal size_of_output
    Return :
        div_array : numpy array of numbers of 1,2,4
    """
    div_array = np.zeros([dim1, dim2])  # make division array
    one_array = np.ones([crop_size, crop_size])  # one array to be added to div_array
    dim1_stride = stride_size(dim1, crop_num1, crop_size)  # vertical stride
    dim2_stride = stride_size(dim2, crop_num2, crop_size)  # horizontal stride
    for i in range(crop_num1):
        for j in range(crop_num2):
            # add ones to div_array at specific position
            div_array[dim1_stride*i:dim1_stride*i + crop_size,
                      dim2_stride*j:dim2_stride*j + crop_size] += one_array
    return div_array


def image_concatenate(image, crop_num1, crop_num2, dim1, dim2):
    """concatenate images
    Args :
        image : output images (should be square)
        crop_num2 (int) : number of crop in horizontal way (2)
        crop_num1 (int) : number of crop in vertical way (2)
        dim1(int) : vertical size of output (512)
        dim2(int) : horizontal size_of_output (512)
    Return :
        div_array : numpy arrays of numbers of 1,2,4
    """
    crop_size = image.shape[1]  # size of crop
    empty_array = np.zeros([dim1, dim2]).astype("float64")  # to make sure no overflow
    dim1_stride = stride_size(dim1, crop_num1, crop_size)  # vertical stride
    dim2_stride = stride_size(dim2, crop_num2, crop_size)  # horizontal stride
    index = 0
    for i in range(crop_num1):
        for j in range(crop_num2):
            # add image to empty_array at specific position
            empty_array[dim1_stride*i:dim1_stride*i + crop_size,
                        dim2_stride*j:dim2_stride*j + crop_size] += image[index]
            index += 1
    return empty_array
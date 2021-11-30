import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage.filters import laplace

def calc_gradients_test(test_dir):
    for i in range(24):
        calc_gradients(test_dir + '/test{}'.format(i))

def calc_gradients(dir):
    g_noisy_dir = dir + '/g_noisy.png'
    p_noisy_dir = dir + '/p_noisy.png'
    g_noisy = Image.open(g_noisy_dir)
    g_noisy = np.asarray(g_noisy)
    p_noisy = Image.open(p_noisy_dir)
    p_noisy = np.asarray(p_noisy)
    g_noisy_grad = gradients(g_noisy)
    p_noisy_grad = gradients(p_noisy)
    Image.fromarray(g_noisy_grad).save(dir + '/g_noisy_grad.png')
    Image.fromarray(p_noisy_grad).save(dir + '/p_noisy_grad.png')


def gradients(img):
    """Compute the xy derivatives of the input buffer. This helper is used in the _preprocess_<base_model>(...) functions
    Args:
        buf(np.array)[h, w, c]: input image-like tensor.
    Returns:
        (np.array)[h, w, 2*c]: horizontal and vertical gradients of buf.
    """
    # dx = img[:, 1:, ...] - img[:, :-1, ...]
    # dy = img[1:, ...] - img[:-1, ...]
    # dx = np.pad(dx, [[0, 0], [1, 0], [0, 0]], mode="constant") # zero padding o the left
    # dy = np.pad(dy, [[1, 0], [0, 0], [0, 0]], mode='constant')  # zero padding to the up
    # dx = sobel(gaussian_filter(img, 31), axis=0, mode='nearest')
    # dy = sobel(gaussian_filter(img, 31), axis=1, mode='nearest')
    dx = laplace(gaussian_filter(img, 10))
    
    return dx
    
        
        
# calc_gradients('test/kpcn_decomp_mask_2/test5')
calc_gradients_test('test/kpcn_decomp_mask_2')
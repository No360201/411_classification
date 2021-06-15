import cv2
import numpy as np


def get_img_np_nchw(filename,max_batch_size):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (224, 224))
    miu = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = np.array(image_cv, dtype=np.float)/255.
    img_np = img_np.transpose((2, 0, 1))
    img_np -= miu
    img_np /= std
    img_np_nchw = img_np[np.newaxis]
    img_np_nchw = np.tile(img_np_nchw,(max_batch_size, 1, 1, 1))
    return img_np_nchw
import struct

import numpy as np
from PIL import Image


""" Preprcoss imgnet to match with tflite format(NHWC)
then pack it as .bin
data layout : (1, 224, 224, 3)
can feed as dataset for model like : Mobilenet_V1_1.0_224_quant
"""


def get_datas_path(path, converted_size, start):
    datas_path = []
    for i in range(converted_size):
        i += start
        num = str(i + 1).zfill(4)
        name = path + '/ILSVRC2012_val_0000' + num + '.JPEG'
        datas_path.append(name)
    return datas_path


def preprocess(image_path):
    image = Image.open(image_path).resize((224, 224))
    x = np.array(image).astype('uint8')

    # wrong format (may happen if C == 1)
    if len(x.shape) != 3:
        return 0, False

    image_data = np.reshape(x, (1, 224, 224, 3))
    return image_data, True


if __name__ == "__main__":
    start = 0
    converted_size = 10
    img_dir =               # path to your imagenet dir (with .jpeg)

    datas_path = get_datas_path(img_dir, converted_size, start)

    preprocessed_imgs = []
    for i in range(converted_size):
        preprocessed_img, success = preprocess(datas_path[i])
        if success:
            b_datas = preprocessed_img.tobytes()
            i += start
            num = str(i + 1).zfill(4)
            with open('tflite_quant_ILSVRC2012_val_0000' + num + '.bin', 'wb') as f:
                f.write(b_datas)

            preprocessed_imgs.append(preprocessed_img)
        else:
            print('get wrong format of img : ', datas_path[i])

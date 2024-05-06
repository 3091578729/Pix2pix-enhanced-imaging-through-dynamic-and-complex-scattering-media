import cv2
import os
import numpy as np
import glob

root = '.\\mnist_images'
for i in range(0,10):
    num = str(i)
    bmp_files = sorted(glob.glob(os.path.join(root, num) + "/*.bmp"))
    png_file = sorted(glob.glob(os.path.join(root, num) + "/*.png"))
    label = cv2.imread(png_file[0])
    print(label.shape)
    label = cv2.resize(label,(256,256))
    for j,image_path in enumerate(bmp_files):

        image = cv2.imread(image_path)
        image = cv2.resize(image,(256,256))
        concate = np.concatenate((image,label),axis=1)
        result_name = './mnist/' + str(i) + '_' + str(j) + '.jpg'
        cv2.imwrite(result_name,concate)

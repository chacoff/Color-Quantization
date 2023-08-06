import sys
import numpy as np
import cv2
import timeit
import matplotlib.pyplot as plt
from functools import lru_cache
from math import sqrt


@lru_cache()
def kmeans_main(**kwargs):

    centers = [[high, high, high], [low, low, low]]
    K = len(centers)

    # Initializing class and distance arrays
    classes = np.zeros([IMAGE_3D_MATRIX.shape[0], IMAGE_3D_MATRIX.shape[1]], dtype=np.float64)
    distances = np.zeros([IMAGE_3D_MATRIX.shape[0], IMAGE_3D_MATRIX.shape[1], K], dtype=np.float64)

    for i in range(10):
        # finding distances for each center
        for j in range(K):
            distances[:, :, j] = np.sqrt(((IMAGE_3D_MATRIX - centers[j]) ** 2).sum(axis=2))

        # choosing the minimum distance class for each pixel
        classes = np.argmin(distances, axis=2)

        # rearranging centers
        for c in range(K):
            centers[c] = np.mean(IMAGE_3D_MATRIX[classes == c], 0)

    # changing values with respect to class centers
    for i in range(IMAGE_3D_MATRIX.shape[0]):
        for j in range(IMAGE_3D_MATRIX.shape[1]):
            IMAGE_3D_MATRIX[i][j] = centers[classes[i][j]]

    # plt.close('all')
    cv2.imwrite('output.png', IMAGE_3D_MATRIX)
    # plt.imshow(IMAGE_3D_MATRIX, cmap='gray')
    # plt.show()
    # return IMAGE_3D_MATRIX


def BorderDetectionCanny(border_channel, canvas):
    im_bw = cv2.threshold(border_channel, 25, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('', im_bw)
    cv2.waitKey(0)

    canny = cv2.Canny(im_bw, 0, 20)  # for border detection
    # [0] instead of using hierarchy for findContours()
    contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # [max(contour, key=cv2.contourArea)]
    hull = []
    for i in range(len(contours)):
        hu = cv2.convexHull(contours[i], returnPoints=False)
        hull.append(hu)
        # plt.plot(hu)
        # plt.show()

    for j in range(len(contours)):
        filled_contour = cv2.drawContours(canvas, contours, j,  (255, 255, 255), thickness=cv2.FILLED)

    binaryfromcnt = cv2.split(filled_contour)[:3][0]  # splitting in RGB channels

    return binaryfromcnt


if __name__ == "__main__":

    # Parameters
    PATH_TO_FILE = f"Test_Inputs\\02_mod.bmp"
    high = 20  # high threshold in grey levels
    low = 10  # low threshold in grey levels
    ROI_start_y = 900  # ROI start point in y direction
    ROI_region = 800  # ROI window

    IMAGE = cv2.imread(PATH_TO_FILE, cv2.IMREAD_UNCHANGED)
    IMAGE = IMAGE[ROI_start_y:(ROI_start_y+ROI_region), 0:IMAGE.shape[1]]  # y:y+h, x:x+w

    IMAGE_int = np.array(IMAGE).astype(int)
    print(f'{IMAGE_int.shape = }')

    IMAGE_3D_MATRIX = cv2.merge((IMAGE_int, IMAGE_int, IMAGE_int)) if len(IMAGE_int.shape) < 3 else IMAGE_int
    image_original = IMAGE_3D_MATRIX.copy()  # just to save it
    print(f'{IMAGE_3D_MATRIX.shape = }')

    execution_time = timeit.timeit(f'{kmeans_main()}') # remember, the function has to be void
    sys.exit(f"{1000*execution_time:0.3f} [s]")

    '''
    image_out = kmeans_main().astype(np.uint8)

    scale_percent = 60  # percent of original size
    width = int(image_out.shape[1] * scale_percent / 100)
    height = int(image_out.shape[0] * scale_percent / 100)
    dim = (width, height)

    image_out = cv2.resize(image_out, dim, interpolation=cv2.INTER_AREA)

    r, g, b = cv2.split(image_out)[:3]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=1)
    g = cv2.dilate(g, kernel, iterations=1)

    to_show = BorderDetectionCanny(g, r)
    cv2.imshow('', to_show)
    cv2.waitKey()
    '''




import sys
import numpy as np
import cv2
import timeit


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


if __name__ == "__main__":

    # Parameters
    PATH_TO_FILE = f"Test_Inputs\\02_24.bmp"
    high = 20  # high threshold in grey levels
    low = 10  # low threshold in grey levels
    ROI_start_y = 900  # ROI start point in y direction
    ROI_region = 800  # ROI window

    IMAGE = cv2.imread(PATH_TO_FILE, cv2.IMREAD_UNCHANGED)
    IMAGE = IMAGE[ROI_start_y:(ROI_start_y+ROI_region), 0:IMAGE.shape[1]]  # y:y+h, x:x+w
    IMAGE_int = np.array(IMAGE).astype(int)
    print(f'{IMAGE_int.shape = }')

    IMAGE_3D_MATRIX = cv2.merge((IMAGE_int, IMAGE_int, IMAGE_int)) if len(IMAGE_int.shape) < 3 else IMAGE_int
    print(f'{IMAGE_3D_MATRIX.shape = }')

    execution_time = timeit.timeit(f'{kmeans_main()}')
    sys.exit(f"{1000*execution_time:0.3f} [s]")

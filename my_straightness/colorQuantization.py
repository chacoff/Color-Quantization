import numpy as np
import cv2

def kmeans_main(IMAGE_3D_MATRIX,high,mid,low):

    centers = [[high, high, high], [mid, mid, mid], [low, low, low]]
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

    return IMAGE_3D_MATRIX


def BorderDetectionCanny(g,drawing):
    # The problem with the test images is, the beam is not closed so the contour will not be closed!
    im_bw = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    canny = cv2.Canny(im_bw,0,20)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cont_sorted = sorted(contours, key=cv2.contourArea, reverse = True)
    # hull_big = [cv2.convexHull(cont_sorted[0],False)]

    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(cont_sorted[i], False))

    for i in range(len(contours)):
    # color_contours = (255, 255, 255) # green - color for contours
    # color = (0, 255, 0) # blue - color for convex hull

        cv2.drawContours(drawing, cont_sorted, i, (0,0,255), 1)
        # cv2.drawContours(drawing, hull, i, (0,255,0), 1)
        # print(f'{len(hull) = }, {hull[0].shape = }')
    return drawing
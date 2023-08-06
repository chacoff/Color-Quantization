from straightness import *
from colorQuantization import *
import time
from multiprocessing import Pool


class finalRun:
    def __init__(self):
        self.images = read_images()
        self.images_unchanged = read_images_unchanged()
        self.high = 27  # high threshold in grey levels
        self.mid = 22  # mid interest point of gray level
        self.low = 17  # low threshold in grey levels
        self.ROI_start_y = 900  # ROI start point in y direction
        self.ROI_region = 1100  # ROI along y axis
        self.lst = [i for i in range(len(self.images))]
        self.pwd = os.path.dirname(os.path.abspath(__file__))

    def getBorder(self,i):
        # start = time.time()
        image_out = kmeans_main(self.images_unchanged[i],self.high,self.mid,self.low).astype(np.uint8)
        # end = time.time()
        # print(f'Time taken to run kmaens_main for image {i}: {end - start}')
        scale_percent = 60  # percent of original size
        width = int(image_out.shape[1] * scale_percent / 100)
        height = int(image_out.shape[0] * scale_percent / 100)
        dim = (width, height)
        im = cv2.resize(self.images[i], dim, interpolation=cv2.INTER_AREA)
        image_out = cv2.resize(image_out, dim, interpolation=cv2.INTER_AREA)

        r, g, b = cv2.split(image_out)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=1)
        # start = time.time()
        to_show = BorderDetectionCanny(g,im)
        # end = time.time()
        # print(f'Time taken to run BorderDetectionCanny for image {i}: {end - start}')
        cv2.imwrite(f'{self.pwd}\\output\\image_out_{i}.jpg',to_show)

    def runParallel(self):
        start = time.time()
        pool = Pool(processes=len(self.images))
        pool.map(self.getBorder,range(0,len(self.images)))
        end = time.time()
        print(f'Time taken to run runParallel for all images: {end - start}')


if __name__ == "__main__":
    
    k = finalRun()
    k.runParallel()

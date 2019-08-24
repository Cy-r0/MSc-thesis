import math
import multiprocessing
import os
import timeit

import cv2
import numpy as np

from config.config import VOCConfig


cfg = VOCConfig()

class DistanceTransform(object):
    """
    Take a directory of pictures and generate a    new directory
    with distance transformed pictures.
    The distance transform converts the image to a representation where
    all the pixel values represent the distance of that pixel from the closest
    boundary pixel.

    Args:
        - img_dir (string): path of directory with instance segmentation images.
        - imageset_f (string): path of file that contains list of images to convert.
        - transformed_dir (string): path of directory with transformed images.
        - img_extension (string, optional): default is .png,
            which is compatible with the PASCAL VOC images.
    """

    def __init__(self,
        img_dir,
        imageset_f,
        transformed_dir,
        img_extension=".png"):
        self.img_dir = img_dir
        self.imageset_f = imageset_f
        self.transformed_dir = transformed_dir
        self.img_ext = img_extension

        if not os.path.isdir(img_dir):
            raise RuntimeError("Image directory not found or corrupted.")
        if not os.path.exists(imageset_f):
            raise RuntimeError("Imageset file not found.")
        
        # Get image names
        with open(os.path.join(imageset_f), "r") as f:
            self.images = [x.strip() for x in f.readlines()]
        
        assert len(self.images) > 0, "No images were retrieved."


    def generate(self,
        n_jobs=0):
        """
        Generate transformed images.

        Args:
            - n_jobs (int): number of processes to spawn. If 0, no
                multiprocessing will be used.
        """
        start = timeit.default_timer()

        if n_jobs > 0:
            pool = multiprocessing.Pool(processes=n_jobs)
            for image in self.images:
                pool.apply_async(self._work,
                                args=(image,))
            pool.close()
            pool.join()

        elif n_jobs == 0:
            for image in self.images:
                self._work(image)

        else:
            print("Invalid number of jobs. Aborted.")


        elapsed = timeit.default_timer() - start

        print("Finished! Time taken: %.2f s (%.2f min).\n"
              %(elapsed, elapsed/60))


    def _work(self, image_name):
        """
        Internal function that converts images.

        Args:
            - image_name (string): name of input image.
        """
        # skip if already exists
        if os.path.exists(os.path.join(self.transformed_dir,
                                        image_name + self.img_ext)):
            pass

        else:
            img = cv2.imread(os.path.join(self.img_dir, image_name + self.img_ext))

            # Get rid of unlabelled regions in the image
            # by creating a mask of the non-unlabelled regions
            # and applying it to the original image 
            unlabelled_mask = cv2.inRange(
                img,
                np.array([192,224,224]),
                np.array([192,224,224])) # Remember OpenCV uses BGR not RGB
            unlabelled_mask = cv2.bitwise_not(unlabelled_mask)
            img = cv2.bitwise_and(img, img, mask=unlabelled_mask)

            contoured = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")

            # Get all unique colours in the image
            colours = np.unique(img.reshape(-1, img.shape[2]), axis=0)
            for c in colours:

                # Ignore background colour
                if np.array_equal(c, np.array([0,0,0])):
                    continue

                else:
                    # Mask single colour in image
                    c_mask = cv2.inRange(img, c, c)
                    # Dilate mask by 2 pixels, which is the average 
                    # halfwidth of the unlabelled ring
                    kernel = np.ones((5,5),np.uint8)
                    c_mask = cv2.dilate(c_mask, kernel, iterations=1)
                    # Find all contours and accumulate them
                    _, contours, _ = cv2.findContours(
                        c_mask,
                        cv2.RETR_LIST, 
                        cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(contoured, contours, -1, 255, 1)
            
            # Invert contoured image because cv2.distanceTransform calculates
            # distance from nearest black region
            contoured = cv2.bitwise_not(contoured)

            distance = cv2.distanceTransform(contoured, cv2.DIST_L2, 5)
            # distance needs to be clipped to 255 and set to uint
            distance = np.clip(distance, 0, 255).astype("uint8")

            cv2.imwrite(os.path.join(
                self.transformed_dir,
                image_name + self.img_ext), distance)
        
        print("Progress: %d/%d images..."
              %(len(os.listdir(self.transformed_dir)), len(self.images)),
              end="\r", flush=True)


if __name__ == "__main__":

    root = "/home/cyrus/Datasets/VOCdevkit/VOC2012/"

    w = DistanceTransform(root + "SegmentationObject",
                            root + "ImageSets/Segmentation/trainval.txt",
                            root + "DistanceTransform")
    w.generate(n_jobs=16)
# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, width, height):
        # store the target's width and height
        self.width = width
        self.height = height

    def preprocess(self, image):
        # extract the random crop from the original image
        # with the target width and height
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]

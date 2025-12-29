import cv2

def filter(image, h=10, templateWindowSize=7, searchWindowSize=21):

    if len(image.shape) == 2:
        # Grayscale
        return cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)
    else:
        # Color
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)
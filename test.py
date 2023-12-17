import cv2
import modules.Initialize
import matplotlib.pyplot as plt
path = "./img/1.jpg"

img = cv2.imread(path)

def adjust_contrast(img, alpha=1.0, beta=0):
    """
    Adjusts the size, contrast, and brightness of an image.
    
    :param img: The original image.
    :param alpha: The contrast control parameter. The higher the value, the higher the contrast. Default is 1.0.
    :param beta: The brightness control parameter. The higher the value, the higher the brightness. Default is 0.
    :return: The adjusted image.
    """
    if img.shape[0] > 2000:
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

# Normalize the image
normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

b, g, r = cv2.split(img)

img_bg = cv2.merge([b, g, 0 * r])

normalized_img = adjust_contrast(img, 1.6)

cv2.imshow("img", img)

cv2.imshow("adj", normalized_img)
cv2.waitKey(0)

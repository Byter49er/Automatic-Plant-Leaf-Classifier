import cv2
from skimage.io import imshow, imread
from matplotlib import pyplot as plt

class leaf:


    def __init__(self,image,featureVector,species):
        self.image = image
        self.featureVector = featureVector
        self.species = species

    def getFeatures(self):
        return self.featureVector


    def getSpecies(self):
        return self.species

    def getImage(self):
        return self.image

    def showImage(self):
        imshow(self.image)
        plt.show()

    def preprocessing(self,path,dispImage):
        self.image = cv2.imread(path)
        if dispImage:
            self.showImage()

        self.image = cv2.resize(self.image, (300, 300), interpolation=cv2.INTER_AREA)

        self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

        self.image = cv2.blur(self.image, (7, 7))




    def segmentation(self):

        T, self.image = cv2.threshold(self.image, self.image.mean()/2,255,cv2.THRESH_BINARY_INV)
        #self.showImage()



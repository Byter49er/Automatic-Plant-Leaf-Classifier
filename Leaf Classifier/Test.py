from Leaf import leaf
import pickle
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
import math
import os

def extractFeatures(newLeaf):
    df = pd.DataFrame()
    label_img = label(newLeaf.getImage())
    table = pd.DataFrame(regionprops_table(label_img, newLeaf.getImage(),
                                           ['convex_area', 'area', 'centroid',
                                            'eccentricity', 'extent',
                                            'inertia_tensor',
                                            'major_axis_length',
                                            'minor_axis_length', 'perimeter', 'solidity', 'image',
                                            'orientation',
                                            'moments_central', 'moments_hu', 'euler_number',
                                            'equivalent_diameter',
                                            'mean_intensity', 'bbox', 'feret_diameter_max']))

    table['perimeter_area_ratio'] = (table['perimeter']) / table['area']


    circularity = []
    for prop in regionprops(label_img):
        if prop.perimeter > 0:
            circularity += [4 * ((prop.area * math.pi) / (prop.perimeter * prop.perimeter))]
        else:
            circularity += [0]

    table['circularity'] = circularity
    df = pd.concat([df, table], axis=0)

    X = df.drop(columns=['image'])

    X = X[['feret_diameter_max','convex_area', 'area', 'inertia_tensor-1-1' ,'major_axis_length', 'minor_axis_length', 'perimeter', 'solidity' ,'moments_hu-0', 'moments_hu-1', 'moments_hu-2', 'moments_hu-3', 'moments_hu-4', 'moments_hu-5', 'moments_hu-6', 'euler_number', 'perimeter_area_ratio','circularity']]  # features

    return X

image_dir_list = os.listdir("LeafDataset/test")
model = pickle.load(open('trainedModel.sav','rb'))
#species = ['abies_concolor','abies_nordmanniana','acer_campestre','acer_ginnala','acer_negundo','acer_palmatum','acer_pensylvanicum','carya_tomentosa','carya_ovata','pinus_bungeana']
correctClassCount = 0
targetClassName = ""
oneOrAll = ""
while not(oneOrAll.lower() == 'yes' or oneOrAll.lower() == 'no'):
 oneOrAll = input("Classify all test leaves?(enter yes or no only)\n>")


if oneOrAll == "yes":
    print("Classifying all leaves")
    for i in range(len(image_dir_list)):
        if(not (image_dir_list[i] == "Thumbs.db")):
         images = os.listdir("LeafDataset/test/"+image_dir_list[i])
         targetClassName = image_dir_list[i]
         print("Target Class Name: "+targetClassName)
        else:
            continue
        for j in range(len(images)):
         if (not(images[j] == "Thumbs.db")):
           testLeaf = leaf([], [], "")

           testLeaf.preprocessing("LeafDataset/test/"+image_dir_list[i]+"/"+images[j],False)
           testLeaf.segmentation()

           features = extractFeatures(newLeaf=testLeaf)
           predictedClass = model.predict(features)[0]
           print("Image "+images[j]+" classified as: " + str(predictedClass))
           if str(predictedClass) == targetClassName:
               correctClassCount= correctClassCount+1
        print()

    print(str(correctClassCount)+" correct classifications out of 50")
    print(str((correctClassCount/50)*100)+"% test accuracy")
else:
    print("Classifying one leaf only:")
    imageName = input("Enter the full path of a leaf image from the test folder that you want to classify:\n>")
    testLeaf = leaf([], [], "")
    try:
       testLeaf.preprocessing(imageName,True)
    except AttributeError as ex:
        print("Could not find the image file specified")
        exit(-1)
    testLeaf.segmentation()
    features = extractFeatures(newLeaf=testLeaf)
    predictedClass = model.predict(features)[0]
    print("The predicted class of this leaf is: "+predictedClass)


import os
import pickle
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.neighbors import KNeighborsClassifier
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import numpy as np
from Leaf import leaf
import math
import time



def main():
    image_dir_list = os.listdir("LeafDataset")
    image_dir_list.remove('test')
    df = pd.DataFrame()
    print("Reading training data and extracting features...")
    for i in range(len(image_dir_list)):
        image_paths = os.listdir("LeafDataset/" +image_dir_list[i])
        for j in range(len(image_paths)):
            if(not(image_paths[j]=="Thumbs.db")):

                newLeaf = leaf([], [], "")
                newLeaf.preprocessing("LeafDataset/" + image_dir_list[i] + "/" + image_paths[j],False)

                newLeaf.segmentation()


                label_img = label(newLeaf.getImage())
                table = pd.DataFrame(regionprops_table(label_img,newLeaf.getImage(),
                                                       ['convex_area', 'area','centroid',
                                                        'eccentricity', 'extent',
                                                        'inertia_tensor',
                                                        'major_axis_length',
                                                        'minor_axis_length', 'perimeter', 'solidity', 'image',
                                                        'orientation',
                                                        'moments_central', 'moments_hu', 'euler_number',
                                                        'equivalent_diameter',
                                                        'mean_intensity', 'bbox','feret_diameter_max']))


                table['perimeter_area_ratio'] = (table['perimeter'])/table['area']



                table['label'] = image_dir_list[i]

                circularity = []
                for prop in regionprops(label_img):
                    if prop.perimeter>0:
                      circularity += [4 * ((prop.area * math.pi) / (prop.perimeter * prop.perimeter))]
                    else:
                      circularity+= [0]



                table['circularity'] = circularity
                df = pd.concat([df, table], axis=0)




    X = df.drop(columns=['label', 'image'])



    X = X[['feret_diameter_max','convex_area', 'area', 'inertia_tensor-1-1' ,'major_axis_length', 'minor_axis_length', 'perimeter', 'solidity' ,'moments_hu-0', 'moments_hu-1', 'moments_hu-2', 'moments_hu-3', 'moments_hu-4', 'moments_hu-5', 'moments_hu-6', 'euler_number', 'perimeter_area_ratio','circularity']] # features

    y = df['label']  #labels



    print("Training model...")
    model = RandomForestClassifier(n_estimators=100,max_depth=20,random_state=123)#classify with Random Forests


    model.fit(X, y)#train model


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=123,
                                                       stratify=y,)  # obtaining validation sets

    print(classification_report(model.predict(X_val), y_val))  # print confusion matrix of validation set
    print(f"Training Accuracy: {np.mean(model.predict(X_val) == y_val) *100:.2f}%"  )  # print training accuracy
    pickle.dump(model, open('trainedModel.sav', 'wb'))  # save trained model

if __name__== '__main__':
    main()

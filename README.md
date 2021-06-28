Author: Ricardo Aaron Pillay

Instructions to Run:

1)Download the "Leaf Classifier" folder

2)Open the "Leaf Classifier" folder and extract the "LeafDataset" folder from the "LeafDataset.zip" file. Note the program will not run if the "LeafDataset.zip"
  is not unzipped prior to running.

3)Open the the "Leaf Classifier" folder in a Python IDE of your choice (This project was developed on PyCharm)

4)There are 2 files that run, namely: "TrainModel.py" "Test.py." and See step 5 for instructions for running "TrainModel.py" and see step 6-7 for instructions 
  for running "Test.py." Note that the model for classification is already trained, so it is ready to be tested immediately once downloaded.
     
5)When "TrainModel.py" is run, there are no required inputs. It will read the training images and extract their features, thereafter training the model
  and finally display a training data accuracy report, using a validation set.

6)When "Test.py" is run, you will be asked: "Classify all test leaves?" If you type "no" to the question, the program will allow you to classify 
  a leaf of your choice from the "test" folder. Copy and paste the path of the leaf image from the "test" folder that you want to test. The specified leaf will be         displayed and once you close the image, the predicted classification will be displayed in the console.

7)When "Test.py" is run, you will be asked: "Classify all test leaves?" If you type "yes" to the question, the classifier will classify all leaves in the test folder       and display the test accuracy percentage afterwards. No images will be displayed.



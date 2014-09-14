import numpy as np
from skimage.io import imread
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import TanhLayer

category = 0

shapes = ["rectangle", "triangle", "circle"]


ds_training = ClassificationDataSet(1024, nb_classes=3, class_labels = ["rectangle", "triangle", "circle"])
ds_testing = ClassificationDataSet(1024, nb_classes=3)


for shape in shapes:
    for i in range(15):
        image = imread('C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/shapes/training/'+shape+str(i+1)+'.png', as_grey=True, plugin=None, flatten=None)
        image_vector = image.flatten()
        ds_training.appendLinked(image_vector, [category])
    category+=1

category = 0

for shape in shapes:
    for i in range(8):
        image = imread('C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/shapes/testing/'+shape+str(i+1)+'.png', as_grey=True, plugin=None, flatten=None)
        image_vector = image.flatten()
        ds_testing.appendLinked(image_vector, [category])
       


   
ds_training.calculateStatistics()
ds_training.getClass(0)
print(ds_training.getField('target'))

ds_training._convertToOneOfMany(bounds=[0, 1])
ds_testing._convertToOneOfMany(bounds=[0, 1])
print(ds_training.getField('target'))
        


net = buildNetwork(1024,12, 12, 3, hiddenclass = TanhLayer, outclass=SoftmaxLayer)
trainer = BackpropTrainer(net, dataset=ds_training, verbose=True, learningrate=0.01)


trainer.trainUntilConvergence()

out = net.activateOnDataset(ds_testing)
out = out.argmax(axis=1) 
print(out) # the highest output activation gives the class
out = out.reshape(X.shape)  
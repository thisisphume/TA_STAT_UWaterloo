# (Tutorial Seesion) Detecting the COVID-19 cases from chest X-ray images.
The machine learning-based model must detect whether the subject of study has been infected or not.

# Data 
This dataset consists of 15264 (512x512) chest X-ray images for the train set and 400 images for the test set. The dataset contains positive and negative classes to indicate the positive and negative COVID-19 cases. The dataset can be downloaded from the class's respository. 

# Approach
Developed the classification model using transfer learning & fine-tuning techniques via two different framework, **FastAI** and **Keras**.

# Key Observations
- The most time consuming step is data preparation. 
- Transfer learning in FastAI is much easier than Keras. We just need to call `fine_tune()` function. 
In Keras, it is a bit more complicated. You need to manually add the classifier layer, train the last layer (freeze the base model), and then train the whole model.
However, it is more flexible in Keras. Some of the FastAI object is just a Pytorch obejct so it can be integrated it with Pytorch.
- FastAI's `lr_find()` is really handy. It lets us know which learning rate should we use. The learning rate is one of the most important hyperparameter.
- Based from the experiment, Keras took way longer to train the model and quite a few epoch for the model to converge.
- The performance (F1-macro) between weighted loss funciton and loss function are much different. The weighted loss function is achieved via `class_weight` option in Keras. See Keras notebook.


# Note on Data Pipline in Tensorflow-Keras 2.x
3 Different Approaches:
1. Build a data pipeline via numpy or tf.tensor. 
    - Using CV2 or PIL library to store the data into numpy, called it X.
    - Build a label in array, called it y
    - Fit the model via `model.fit(X,y)` command.
    - Pro: Easy to understand / Straigth forward / good for small dataset
    - Con: Use a lot of mememory (data is stored in the memory) 

2. Using data generator class from `tf.keras.utils.Sequence`
    - Build a generator class that will store the data batch by batch. 
    - Built-in data augmentation fuctnion and can be use with albumentations and nlpaug library.

3. Using `tf.data.dataset` 
    - Load the data, data augmentation
    - Faster than other 2 methods because it has been carefully optimized. 
    - Example: [Kaggle: RFCX: train resnet50 with TPU](https://www.kaggle.com/yosshi999/rfcx-train-resnet50-with-tpu?fbclid=IwAR2DGIoJE0V0vG7oCOXXQwmFqG7ykcjzp4TKLBTLv2D7dbWFVyN8OwyfR98)


# Pre-trained Model
EfficientNetB0 / DenseNet / VGG / XCeption
- [Best deep CNN architectures and their principles: from AlexNet to EfficientNet](https://theaisummer.com/cnn-architectures)

| Model name | Number of parameters [Millions] |	ImageNet Top 1 Accuracy	| Year |
| --- | --- | --- | --- |
| AlexNet |	60 M |	63.3 % |	2012
| Inception V1 |	5 M |	69.8 % |	2014
| VGG 16  |	138 M |	74.4 % |	2014
| VGG 19 |	144 M |	74.5 % |	2014
| Inception V2 |	11.2 M |	74.8 % |	2015
| ResNet-50 |	26 M |	77.15 % |	2015
| ResNet-152 |	60 M |	78.57 % |	2015
| Inception V3 |	27 M |	78.8 % |	2015
| DenseNet-121 |	8 M |	74.98 % |	2016
| DenseNet-264 |	22M |	77.85 % |	2016
| BiT-L (ResNet) |	928 M |	87.54 % |	2019
| NoisyStudent EfficientNet-L2 |	480 M |	88.4 % |	2020
| Meta Pseudo Labels |	480 M |	90.2 % |	2021


# Ref: 
- [7 Reasons why you should ABSOLUTELY know about fast.ai deep learning library](https://medium.com/analytics-vidhya/7-reasons-why-you-should-absolutely-know-about-fast-ai-deep-learning-library-890cf4e293de)
- [Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/)
- @thaikeras
- [Keras Image data preprocessing](https://keras.io/api/preprocessing/image/?fbclid=IwAR2-LLBGMGr0gIp1huUcStrJOxFnNqx-0p69LF6nBgewgj8ApBGRgUODYWg#imagedatagenerator-class)
- [Inside TensorFlow: tf.data + tf.distribute](https://www.youtube.com/watch?v=ZnukSLKEw34)

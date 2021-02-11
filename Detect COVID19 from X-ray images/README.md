# (Tutorial Seesion) Detecting the COVID-19 cases from chest X-ray images.
The machine learning-based model must detect whether the subject of study has been infected or not.

# Data 
This dataset consists of 15264 (512x512) chest X-ray images for the train set and 400 images for the test set. The dataset contains positive and negative classes to indicate the positive and negative COVID-19 cases. The dataset can be downloaded from the class's respository. 

# Approach
Developed the classification model using transfer learning & fine-tuning techniques via two different framework, **FastAI** and **Keras**.

# Note
- The most time consuming step is data preparation. 
- Transfer learning in FastAI is much easier than Keras. We just need to call `fine_tune()` function. 
In Keras, it is a bit more complicated. You need to manually add the classifier layer, train the last layer (freeze the base model), and then train the whole model.
However, it is more flexible in Keras. Some of the FastAI object is just a Pytorch obejct so it can be integrated it with Pytorch.
- FastAI's `lr_find()` is really handy. It lets us know which learning rate should we use. The learning rate is one of the most important hyperparameter.
- Based from the experiment, Keras took way longer to train the model and quite a few epoch for the model to converge.

# Ref: 
- [7 Reasons why you should ABSOLUTELY know about fast.ai deep learning library](https://medium.com/analytics-vidhya/7-reasons-why-you-should-absolutely-know-about-fast-ai-deep-learning-library-890cf4e293de)
- [Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/)

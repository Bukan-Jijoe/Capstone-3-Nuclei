# Capstone-4-Nuclei

## Description
This capstone project is another simple CNN or Computer Vision project to find the cell nuclei based on data from (https://www.kaggle.com/competitions/data-science-bowl-2018/overview) using Tensorflow.Keras. (its my fourth time posting here)
The architecture of this model is a simple CNN model using U-Net. Details will explained further.

## The process of how this model training been done

1. Import Packages
   - Import libraries that might be useful for this model (have some issue for tensorflow.examples.model, just download the library from github and extract it into your env site packages)
![Import Libraries](https://github.com/user-attachments/assets/c638695f-f342-415d-aa4e-b9458c244a60)

2. Data Directory
  - since I don't remember the other way to do data loading for this dataset, I just proceed with simple directory method
![Directory Setting](https://github.com/user-attachments/assets/ad1e227e-d137-4781-9ba1-8fc64292e49f)

3. Data Loading
   - create a load image function where it decode the images into 3 channels and resize the image into our desired size (which is 128 x 128), and normalize it after resize.
   - sames goes to load mask function but it decode into 1 channels only, the rest is same.
   - then we create a load data function that use two previous function to create train_images, train_masks, test_images, test_masks to futher use in data augmentation
![Load Data](https://github.com/user-attachments/assets/96ddf645-9011-40b2-beed-5037b5e73d8c)

4. Augmentation
   - This augmentation function are a custom TensorFlow Keras layer that performs data augmentation on both the input images and their corresponding labels.
   - Specifically, it applies a random horizontal flip to both the input data and labels using TensorFlow's RandomFlip layer.
![Augmentation Function](https://github.com/user-attachments/assets/583012d8-227d-4c0a-9aae-c1a91e673b7a)
   - So we use this augmetation function on train dataset sp that the model exposed to a broader variety of data points, which also increase the amount of training data in images and masks.
![Augmentation Usage](https://github.com/user-attachments/assets/820accc8-fd5b-4ba6-94be-69942adbce2b)
     
5. Display Function
   - This function will be used to in visualizing the images and its mask. Futher down, it will be used to see the predicted mask compare to real mask
![Display Function](https://github.com/user-attachments/assets/6d933e13-619b-4699-a2a3-ff8e0252a73f)

6. Model Transfer Learning
   - For this project, we used a pre trained model from keras.applications (MobileNetV2) as a feature extractor in U-Net Function
![Model Transfer Learning](https://github.com/user-attachments/assets/ed723f8d-9f0d-429a-8a3c-41589939da8b)

8. U_Net Function
   - U-Net is a deep learning architecture primarily designed for image segmentation tasks.
   - Its consists Encoder (Downsampling) and Decoder (Upsampling)
   - Here, the Encoder (Downsampling) or Contracting Path - It reduces spatial resolution while increasing the depth (number of feature channels) to extract high-level features
   - Then, the Decoder (Upsampling) or Expansive Path - where it upscales the feature maps and combines them with corresponding feature maps from the contracting path, allowing the model to recover spatial resolution and make pixel-wise predictions
   - Skip function in this case are bypassing intermediate layers and concatenate feature maps from the contracting path to the corresponding layers in the expansive path. This allows the model to preserve fine-grained spatial information while recovering resolution in      the decoder.
   - Furthermore, U-Net are symmetric architecture which helps the network combine high-level semantic features with low-level spatial details.
![U-Net Down and Up Sampling](https://github.com/user-attachments/assets/000cd648-81dc-49dc-8378-daeb2726d544)
![U-Net Function](https://github.com/user-attachments/assets/1739488a-2085-49a2-8f59-0714ba31af24)

9. DisplayCallBack Function
   - Same as Display function but this is more to show training progress rather than predicting
![DisplayCallBack Function](https://github.com/user-attachments/assets/8590691e-d47c-41cb-8925-a489c9ecd912)

11. Model Callbacks
    - These are my callback function for model training
![Callback Function for Model Training](https://github.com/user-attachments/assets/755e87d4-35ba-4eda-8e89-22eb1760d7bd)
    - As usual, Early Stopping for not making the model overfit, Tensorboard for comparing model performance from previous training, and Displaycallback for showing images, real mask and in-training predicition mask.

13. Model Training
    - We have hyperparameter as shown below (cannot remember why we need it)
![Model Training Setting](https://github.com/user-attachments/assets/f84cfa91-0750-42f3-91ad-a39c48e5304f)
![Model Training Result](https://github.com/user-attachments/assets/e17f076c-c542-4c1d-b3a8-667e33a2034d)
    - This is the accuracy of model
![Epoch Accuracy Model](https://github.com/user-attachments/assets/efce7be3-eb4b-4083-aa58-29637d7709b7)
    - This is the loss of model
![Epoch Loss Model](https://github.com/user-attachments/assets/2325a951-ef5b-4732-8821-d289d9eb89ac)

15. Model Predicition
    - This is the model prediction for test data index number 9 (images and true mask) and predicted mask. Good enough for me.
![Model Prediction](https://github.com/user-attachments/assets/a28e808e-cbc7-4343-98b4-075dce92a41f)

16. Model Save
    - Finally, if we satisfied with the model performance, we can save it like this in h5 format

## Acknowledgements

- Also thanks to my SHRDC Trainers for giving me the opportunity and knowledges about coding, machine learning and deep learning

Step 1- Download dataset of handwritten A-Z characters images in CSV format from  https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format/download 

dataset zip size = 188 MB
unzip file size (in CSV format)= 699 MB

Step 2 - Explore dataset using pandas and numpy, find number of row(or number of images store as their pixel values), and find dimension of downloaded dataset of image.

number of row(images) = 372451
number of comma separated value( number of pixel value) in 1 row = 785

Create work flow daigram. Its link is https://venngage.net/ps/I0hx3ntdyw/handwritten-character-recognition-with-convolutional-neural-network 

step 3 - divide whole set of data into train and test set.
	create 2 csv file train.csv and test.csv from A_Z Handwritten Data.csv using train and test split of sklearn.
	
	
step 4 - Create image from its pixel values in csv file and put them in their respective floders


step 5 - Train my model using 3 layers of CNN and 2 layer of DNN in tensorflow framwork. try 2 model with different number of filters

model_01 - first layer has 16 and second layer has 32 filters which has accuracy = 0.9975, val_accuracy = 0.9753
model_02 - first layer has 32 and second layer has 64 filters which has accuracy = 0.9861, val_accuracy = 0.9838
callback is also implemented with model_02 which stop training when accuracy became equal to 0.985.

create model with 32 and 64 filters in first ans second layers respectively.
get accuary of 99% on training and 95% on validation dataset.

create final model with 32 and 64 filters in first ans second layers respectively.
get accuary of 98% on training and 95% on validation dataset and also using Dropout at DNN layers of where 512 neuron.


step 6 - Create web UI for prediction from model.
	create web interface for uploading an image and get result of predictions.
	show an image which to be predicted.
	show the stages of convolutions.
	
step 7 - deploying interface on AWS cloud take instance of AWS Deep Learning Base AMI (Amazon Linux 2).
	create a content remove python file to delete old image that is save in uploads and static's image folder.
	zip only those file which is usefull in deployment(interface python file, remove old image file, model folder, template, uploads, static floders)

import scipy.misc as im
import numpy as np
import os
# This data preprocessing script needs a lot of matrix operation
# Don't implement them pixel-wisely
# Please search numpy functions online



def load_img_pairs(imgFile,imgLabelFile):
    
    # read in image and label image pairs and formulate into keras shape

	# Step 1: read in image and label image using scipy function imread
	# <img>: since the image is RGB, it has 3 channels,
	# 	   thus the shape is (img_size1 X img_size2 X 3)
	# <imgLabel>: the read in <imgLabelFile> shape is (img_size1 X img_size2)

    img0=im.imread(imgFile)
    img1=im.imresize(img0,(256,256))
    img2=np.asarray(img1,dtype="float32")
    
    labl0=im.imread(imgLabelFile)
    labl1=im.imresize(labl0,(256,256))
    labl2=np.asarray(labl1,dtype="float32")
    
    labl3,labl4=np.unique(labl2,return_inverse=True)
    labl4=np.reshape(labl4,(256,256))
    labl5=np.zeros((256,256,1),dtype="float32")
    for i in range(256):
        for j in range(256):
            if(labl4[i,j]<40):labl5[i,j,0]=0
            elif(labl4[i,j]<130):labl5[i,j,0]=1
            else: labl5[i,j,0]=2
     
    return img2,labl5

# Step 2: formulate imgLabel into 3 channels
	# Step 2(1): <imgLabel> has 3 values, e.g. l_0, l_1, l_2; find them using np.unique()

	# Step 2(2): for each l_i, we create a <bool_label_i> of size (img_size1 X img_size2) using (imgLabel==l0)

	# Step 2(3): label_i = 1*bool_label_i , it can convert bool image into int image
	
	# Step 2(4): concatenate 3 <label_i> into one array <label>
	# 		<label>: shape is img_size1 x img_size2 x 3
	
	
	
	

def load_train_data(dataDir,lablDir):
	# generate your training data
	# read in <num_img> pairs of images and label images, 
	# concatenate them and formulate it into X_train and Y_train
	# <X_train>: shape is (num_img X img_size1 X img_size2 X 3)
	# <Y_train>: shape is (num_img X img_size1 X img_size2 X 3)

	# concat <num_img> images into a list first, then convert to numpy array
	# X_train_list = [img0, img1, img2, ...]
	# X_train = np.asarray(X_train_list)
    imgs=os.listdir(dataDir)
    labls=os.listdir(lablDir)
    num=len(imgs)
    
    for i in range(num):
        imgs[i],labls[i]=load_img_pairs(dataDir+'/'+imgs[i],lablDir+'/'+labls[i])

    data_train=np.asarray(imgs)
    labl_train=np.asarray(labls)

    return data_train,labl_train
dataDir="d:/my code/deep_learning/dis_cup/Training400/imgs"
lablDir="d:/my code/deep_learning/dis_cup/Annotation-Training400/Disc_Cup_Masks/label"
saveDir="d:/my code/deep_learning/dis_cup/"
data_train,labl_train=load_train_data(dataDir,lablDir)
np.save(saveDir+"data.npy",data_train)
np.save(saveDir+"labl.npy",labl_train)
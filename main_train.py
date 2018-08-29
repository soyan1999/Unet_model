from my_model import *
import numpy as np
from keras.utils import np_utils

# where you put your training data
data_al=np.load("d:/my code/deep_learning/dis_cup/data.npy")
data_al=data_al/255
data_al=data_al[0:400]
labl_or=np.load("d:/my code/deep_learning/dis_cup/labl.npy")
labl_al=np_utils.to_categorical(labl_or, 3)
labl_al=labl_al[0:400]
#data_train=data_al[1:400]
#labl_train=labl_al[1:400]
#data_v=data_al[200:220]
#labl_v=labl_al[200:220]
# where you save your trained model
modelFile = "d:/my code/deep_learning/dis_cup/model_9.hdf5"
model_weight_path="d:/my code/deep_learning/dis_cup/model_weight.hdf5"

# Optional: reading images are slow, 
#		it would be better to save X_train and Y_train into an hdf5 file


model = unet()
model.load_weights(model_weight_path)


# compile and fit the model



model.fit(data_al,labl_al,validation_split=0.1,batch_size=1,shuffle=True,epochs=100)


model.save(modelFile)
model.save_weights(model_weight_path)
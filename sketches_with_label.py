import os
import pickle
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
import numpy as np
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
import cv2

import ast
with open('2ksketchWithBox.csv') as f:
    data = f.read()
sketch_with_box = ast.literal_eval(data)

model_file = 'QuickDrawSketch/final_model_sketch.h5'
model = load_model(model_file)

def path_to_array(img_path, shape=(299,299)):
    img = keras_image.load_img(img_path, target_size=shape)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_preprocess_input(img)
    return img

labels_sketch = ['airplane', 'bicycle', 'car', 'cat', 'cloud', 'cow', 'dog', 'elephant', 'fire hydrant', 'giraffe', 'grass', 'horse', 'motorbike', 'sheep', 'traffic light', 'tree', 'zebra']

def load_image_pixels(filename, shape=None, expand_dims_required=False):
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, target_size=shape) 
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0) if expand_dims_required else image
    return image, width, height

def show_image(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()
    
def path_to_array(img_path, shape=(299,299)):
    img = keras_image.load_img(img_path, target_size=shape)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_preprocess_input(img)
    return img
    
class node_image:
    def __init__(self,parent,location,label,score):
        self.parent = parent
        self.location = location
        self.label = label
        self.score = score
        
    def get_image(self, pad=0.20):
        image, width, height = load_image_pixels(self.parent.filename)
        min_x,max_x,min_y,max_y = self.location
        pad_x = int(np.ceil((max_x-min_x)*pad/2));
        pad_y = int(np.ceil((max_y-min_y)*pad/2));
        min_x = max(min_x-pad_x,0)
        max_x = min(max_x+pad_x,len(image[0]))
        min_y = max(min_y-pad_y,0)
        max_y = min(max_y+pad_y,len(image))
        image = image[min_y:max_y,min_x:max_x]
        return image
    
    def predict(self,model):
        image = self.get_image(0.2)*255
        image_shape = image.shape
        image = keras_image.smart_resize(image,size=(299,299))
        image = np.expand_dims(image, axis=0)
        image = inception_preprocess_input(image)
        preds = model.predict(image)[0]
        best_pred_indx = np.argmax(preds)
        self.label = labels_sketch[best_pred_indx]
        self.score = preds[best_pred_indx]
        
    
def merge(box1,box2,mode='mean'):
    xmin = max(box1[0],box2[0])
    xmax = min(box1[1],box2[1])
    ymin = max(box1[2],box2[2])
    ymax = min(box1[3],box2[3])
    area_of_int = (ymax-ymin)*(xmax-xmin)
    if (ymax-ymin)<0 or (xmax-xmin)<0:
        return
    
    box1_area = (box1[1]-box1[0])*(box1[3]-box1[2])
    box2_area = (box2[1]-box2[0])*(box2[3]-box2[2])
    
    if area_of_int/min(box1_area,box2_area) >= 0.5:
        if mode == 'mean':
            return (np.mean([box1[0],box2[0]]).astype('int'),
                    np.mean([box1[1],box2[1]]).astype('int'),
                    np.mean([box1[2],box2[2]]).astype('int'),
                    np.mean([box1[3],box2[3]]).astype('int'))
        else:
            return (np.min([box1[0],box2[0]]).astype('int'),
                    np.max([box1[1],box2[1]]).astype('int'),
                    np.min([box1[2],box2[2]]).astype('int'),
                    np.max([box1[3],box2[3]]).astype('int'))
    return 

class full_image:
    def __init__(self,filename):
        self.filename = filename
        self.shape = cv2.imread(filename).shape
        self.mindim = min(self.shape[0],self.shape[1])
            
    def get_cv_result(self,result_dict):
        if self.filename not in result_dict:
            print('Error no result of {} found in given dict'.format(self.filename))
            return
        boxes = result_dict[self.filename]
        nodes = []
        for box in boxes:
            x,y,w,h = box
            if min(w,h)>self.mindim*0.1:
                location = (x,x+w,y,y+h)
                new_node_image = node_image(self,location,None,None)
                nodes.append(new_node_image)
        N = len(nodes)
        valid = [True]*N       
        for i in range(N):
            for j in range(i):
                if valid[j]:
                    merge_result = merge(nodes[i].location,nodes[j].location)
                    if merge_result is not None:
                        nodes[j]=node_image(self,merge_result,None,None)
                        valid[i]=False
        for i in range(N):
            for j in range(i):
                if valid[j]:
                    merge_result = merge(nodes[i].location,nodes[j].location)
                    if merge_result is not None:
                        nodes[j]=node_image(self,merge_result,None,None)
                        valid[i]=False
        self.nodes = [node for val,node in zip(valid,nodes) if val]
            
        
    def get_yolov3_result(self,result_dict):
        file = self.filename.split('/')[-1]
        res = result_dict[file] if file in result_dict else None
        if res is None:
            print('Error no result of {} found in given dict'.format(self.filename))
            return
        self.nodes = []
        for indx in range(len(res[1])):
            location,labels,scores = res[0][indx],res[1][indx],res[2][indx]
            new_node_image = node_image(self,location,labels,scores)
            self.nodes.append(new_node_image) 
            
    def get_image(self, shape=None):
        image, width, height = load_image_pixels(self.filename, shape)
        return image
    
dataset = {}
for file in sketch_with_box:
    I = full_image(file)
    I.get_cv_result(sketch_with_box)
    for node in I.nodes:
        node.predict(model)
    dataset[file] = I
    if len(dataset)%10==0:
        print(len(dataset))
        
with open('sketch_with_labels.pickle', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
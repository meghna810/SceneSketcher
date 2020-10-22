import os
import pickle

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims

class node_image:
    def __init__(self,parent,location,label,score):
        self.parent = parent
        self.location = location
        self.yolo_label = label
        self.yolo_score = score
        
    def get_image(self, pad=0.40):
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
             
class full_image:
    def __init__(self,filename,result_dict=None):
        self.filename = filename
        if result_dict:
            self.get_yolov3_result(result_dict)
        
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
            
    def get_image(self, shape):
        image, width, height = load_image_pixels(self.filename, shape)
        return image

# load and prepare an image
def load_image_pixels(filename, shape=None, expand_dims_required=False):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape) 
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0) if expand_dims_required else image
    return image, width, height

def show_image(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()

with open('result_withBoxes.pickle','rb') as f:
    result_dict = pickle.load(f)

dir_path = './Datsaset_SketchCOCO/GT/'
dataset = []
for file in result_dict:
    file_path = os.path.join(dir_path,file)
    I = full_image(file_path,result_dict)
    dataset.append(I)
    
print('{} images in dataset'.format(len(dataset)))

#Showing a random node image
z = dataset[174].nodes[0].get_image()
show_image(z)
# SceneSketcher
MLReproducability2020

# Files and usages:
* Pretrained network weights for yolov3 to filter the SketchCOCO dataset obtained from https://pjreddie.com/media/files/yolov3.weights
* create_model_yolov3.py: Saves the yolov3 model by loading the pretrained weights
* get_object_classes.py : Gets the objects present in each image using yolov3 model, and dumps it in  result_final.pickle file 
* pickle_read.py        : Reads a pickle file into a .txt (Usage: python pickle_read.py > result.txt) 
* getMultiObjectJPGs.py : Filters and obtains multi-object images from the SketchCOCO datatset using the result_final.pickle file  
* box1.py               : Gets the bounding boxes coordinates for sketch type images
* word2vec.py           : Finds a vector(category label Ci) representing the corresponding object name 
* 

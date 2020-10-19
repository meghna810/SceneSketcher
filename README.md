# SceneSketcher
MLReproducability2020

# Files and usages:
* getMultiObjectJPGs.py: Used to filter multi-object images from the SketchCOCO datatset
* Pretrained network weights for yolov3 to filter the SketchCOCO dataset obtained from https://pjreddie.com/media/files/yolov3.weights
* create_model_yolov3.py: Saves the yolov3 model by loading the pretrained weights
* get_object_classes.py : Gets the objects present in each image using yolov3 model, and dumps it in  result.pickle file 

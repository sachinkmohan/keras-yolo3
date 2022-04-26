#!/usr/bin/env python
# coding: utf-8

# In[7]:


from yolo import YOLO, detect_video
import cv2


# In[3]:


path_image = cv2.imread('street.jpg')


# In[4]:


#image = cv2.resize(path_image, (416, 416))


# In[5]:


#import matplotlib.pyplot as plt
#plt.imshow(image)


# In[10]:


path_weights='./model_data/yolo.h5'
path_anchors='./model_data/yolo_anchors.txt'
path_classes='./model_data/coco_classes.txt'
nb_gpu=1


# In[11]:


yolo = YOLO(weights_path=path_weights, anchors_path=path_anchors,
                classes_path=path_classes, nb_gpu=nb_gpu)


# In[8]:


image_pred  = yolo.detect_image(path_image)
print('done')


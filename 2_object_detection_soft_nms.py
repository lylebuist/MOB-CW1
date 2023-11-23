#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# # Soft NMS Implementation

# In[2]:


# You will need iou function for implementing Soft-NMS, here is the iou implementation from the Lab2
def iou(box1, box2):
    [box1_x1, box1_y1, box1_x2, box1_y2] = box1
    [box2_x1, box2_y1, box2_x2, box2_y2] = box2
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(0, yi2 - yi1)
    inter_height = max(0, xi2 - xi1)
    inter_area = inter_width * inter_height
    box1_area = (box1_x2 - box1_x1) * ((box1_y2 - box1_y1))
    box2_area = (box2_x2 - box2_x1) * ((box2_y2 - box2_y1))
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou


# #### Please implementing soft NMS algorithm in soft_nms() function

# In[84]:


def soft_nms(scores: np.array, boxes: np.array,sigma=0.5):
    # scores - shape (,n) numpy array that contains scores of bounding boxes
    # boxes - shape (n,4) numpy array that contains bounding boxes information.
    #        Each element in array is a size=4 array of [x1,y1,x2,y2]
    #        Where x1,y1 are the bottom left coordinates of bounding boxes,
    #        and x2,y2 are the top right coordinates of bounding boxes.
    # START of your implementation
    
    new_bbox = np.zeros(shape=(scores.size, 4))
    new_scores = np.zeros(shape=(scores.size))
    counter = 0
    
    while(boxes.size//4 > 0):
        indice_of_max = np.argmax(scores)
        box_m = boxes[indice_of_max]
        
        new_bbox[counter] = box_m
        new_scores[counter] = scores[indice_of_max]
        
        boxes = np.delete(boxes, indice_of_max, 0)
        scores = np.delete(scores, indice_of_max)
        
        for i in range(0, (boxes.size//4)):
            this_iou = iou(box_m, boxes[i])
            scores[i] = scores[i] * np.exp((-this_iou**2)/sigma)
            
        counter += 1
        
    # END of your implementation
    # new_bbox - shape (n,4) numpy array that contains new bounding boxes information
    # new_scores - shape (n,) numpy array that contains new bounding boxes scores
    return new_bbox, new_scores


# In[85]:


boxes = np.array([[200, 200, 400, 400], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400], [1, 1, 2, 2]], dtype=np.float32)
scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)

boxes, scores = soft_nms(scores, boxes, sigma=0.5)
print("scores[2] = " + str(scores[2]))
print("boxes[2] = " + str(boxes[2]))
print("scores = " + str(scores))
print("boxes = " + str(boxes))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))

assert scores.shape == (5,), "Wrong shape"
assert boxes.shape == (5, 4), "Wrong shape"

assert np.isclose(scores[2], 0.31670862), "Wrong value on scores[2]"
assert np.allclose(boxes[2], [220,220,420,420]), "Wrong value on boxes[2]"

assert np.allclose(scores, np.array([0.9,0.5,0.31670862,0.11392745,0.06270898])), "Wrong value on scores"
assert np.allclose(boxes, np.array([[200, 200, 400, 400], [1, 1, 2, 2], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400]])), "Wrong value on boxes"

print("\033[92m All tests passed!")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


# # DICE evaluation metric
# In the lab semantic segmentation, you have implemented IOU to evaluate the performance of the model. Here, you need to implement a similar evaluation metric called DICE or Sørensen–Dice coefficient, and it is formulated as: $$ DICE(X, X_{truth}) = \frac{2|X \cap X_{truth}|}{|X| + |X_{truth}|}$$ \
# Compared to IOU, DICE is more sensitive to small differences in overlap due to the squared terms in the numerator and denominator, so it can be more informative when there's a need to discriminate between segmentations with subtle differences in overlap.

# In[2]:


def DICE(inp : Tensor, tgt : Tensor):
    """
    Arguments:
        inp: Predicted mask (batchsize, number of classes, width, height)
        tgt: Ground truth mask (batchsize, number of classes, width, height)
    Returns:
        Classwise Average of DICE coefficient
    """
    eps = 1e-5 # small number to add to denominator to avoid division by zero
    #YOUR CODE START HERE
    sum_dim = (-1, -2, -3)
    # calculation of intersection   
    inter = 2 *(inp * tgt).sum(dim=sum_dim)

    # calculate the sum of |inp| + |tgt|
    sets_sum = inp.sum(dim=sum_dim) + tgt.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # calcaute the dice    
    dice = (inter + eps) / (sets_sum + eps)
    
    # average the dice batchwise
    return dice.mean()


# ### Tests

# In[3]:


prediction1 = Tensor([[0, 7, 5, 7, 2],
        [2, 4, 5, 9, 9],
        [2, 8, 5, 1, 8],
        [3, 6, 5, 2, 6],
        [3, 2, 9, 1, 1]]).unsqueeze(0).long()
mask1 = Tensor([[4, 2, 5, 0, 2],
        [8, 2, 9, 8, 5],
        [0, 8, 7, 9, 6],
        [8, 6, 5, 9, 1],
        [3, 2, 9, 0, 6]]).unsqueeze(0).long()
prediction2 = Tensor([[5, 7, 3, 3, 0],
        [0, 2, 8, 2, 7],
        [1, 7, 0, 9, 9],
        [7, 5, 2, 3, 4],
        [6, 0, 9, 0, 1]]).unsqueeze(0).long()
mask2 = Tensor([[4, 6, 8, 3, 0],
        [4, 4, 7, 2, 7],
        [0, 0, 4, 9, 9],
        [5, 2, 3, 3, 4],
        [3, 0, 0, 8, 2]]).unsqueeze(0).long()

#Tests
dice1 = DICE(F.one_hot(prediction1).permute(0, 3, 1, 2).float(), F.one_hot(mask1).permute(0, 3, 1, 2).float()).item()
dice2 = DICE(F.one_hot(prediction2).permute(0, 3, 1, 2).float(), F.one_hot(mask2).permute(0, 3, 1, 2).float()).item()

assert np.isclose(0.3200001120567322, dice1), 'incorrect dice 1!'
assert np.isclose(0.3600001037120819, dice2), 'incorrect dice 2!'
print("\033[92m All tests passed!")


# ## Open questions 

# 1. Based on their formulation, what are the **difference** among Cross-entropy, DICE, and IOU?  
# 
#     Cross-entropy is used more often as a proy as a way to make it easier to maximise using back propogation as in  back propagtion knowing the result had large loss the reason as to why can be looked back on and analysed. However DICE and IOU are used to directly maximise metrics. IOU penalises under and over-segmentation more than DICE, meaning that if there are too many or not enough segments when tryring to regognize parts of an image, IOU will be much harsher. In IOU the union is used in the denominator however the union is absent in calculation in DICE. Other than that DICE and IOU are very similar.
#     
# 
# 2. How might the choice of architecture, such as U-Net or FFN, affect the performance and application suitability of your semantic segmentation model?
# 
#     U-Net is able to handle much larger images and create much more accurate image segmentation. If using multiple classes per image U-Net is also preferable. In U-Net architecture, it converts an image into a vector and back again using the same methods

# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:03:40 2021

@author: farzt
"""

import cv2
import numpy as np
#%%
imgs= [r"Image1.jpg", r"Image2.jpg", r"Image3.jpg"]
img = [cv2.imread(fn) for fn in imgs]
ets=np.array([100,250,500],dtype=np.float32)
#%%
mrg=cv2.createMergeDebevec()
#%%
HDR=mrg.process(img,times=ets.copy())
#%%
merg=cv2.createMergeRobertson()
HDRR=merg.process(img,times=ets.copy())
#%%
tm=cv2.createTonemap(gamma=2.2)
res=tm.process(HDR.copy())
M=cv2.createMergeMertens()
rm=M.process(img)
#%%
rd8=np.clip(res*255,0,255).astype("uint8")
rd82=np.clip(rm*255,0,255).astype("uint8")
#%%
cv2.imwrite("HDR1.jpg",rd8)
cv2.imwrite("HDR2.jpg",rd82)




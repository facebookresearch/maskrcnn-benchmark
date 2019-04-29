#!/usr/bin/env python

# --------------------------------------------------------
# LDDP
# Licensed under UC Berkeley's Standard Copyright [see LICENSE for details]
# Written by Samaneh Azadi
# --------------------------------------------------------

import numpy as np

def IoU_target(bbox,gt):
    """compute IoU between bbox and gt
        where both of them are vectors
    """
    M = bbox.shape[0]
    x_1i = bbox[:,0]
    x_1j = gt[:,0]
    x_2i = bbox[:,2]
    x_2j = gt[:,2]
    y_1i = bbox[:,1]
    y_1j = gt[:,1]
    y_2i = bbox[:,3]
    y_2j = gt[:,3]

    w = (np.minimum(x_2i, x_2j) - np.maximum(x_1i, x_1j) + 1)
    h = (np.minimum(y_2i, y_2j) - np.maximum(y_1i, y_1j) + 1)
    w = (w>0) * w
    h = (h>0) * h
    Intersection = w * h
    Area_i = (bbox[:,2] - bbox[:,0] + 1 ) * (bbox[:,3] - bbox[:,1] + 1)
    Area_j = (gt[:,2] - gt[:,0] + 1 ) * (gt[:,3] - gt[:,1] + 1)
    Union = Area_i + Area_j - Intersection
    if np.nonzero(Union ==0)[0].size:
        raise Exception("Union of boxes should not be zero")
    IoU = Intersection/Union

    return IoU

def pair_Intersection(locations):
    """ compute intersection between each pair of boxes in
     the locations matrix
     [x_1i,y_1i,x_2i,y_2i]=locations[i,0:4]
    """
    M = locations.shape[0]
    x_1i = np.reshape(np.repeat(locations[:,0],M),(M,M))
    x_1j = np.reshape(np.tile(locations[:,0],M),(M,M))
    x_2i = np.reshape(np.repeat(locations[:,2],M),(M,M))
    x_2j = np.reshape(np.tile(locations[:,2],M),(M,M))
    y_1i = np.reshape(np.repeat(locations[:,1],M),(M,M))
    y_1j = np.reshape(np.tile(locations[:,1],M),(M,M))
    y_2i = np.reshape(np.repeat(locations[:,3],M),(M,M))
    y_2j = np.reshape(np.tile(locations[:,3],M),(M,M))
    w = (np.minimum(x_2i, x_2j) - np.maximum(x_1i, x_1j) + 1)
    h = (np.minimum(y_2i, y_2j) - np.maximum(y_1i, y_1j) + 1)
    w = (w>0) * w
    h = (h>0) * h
    Intersection = w * h
    return Intersection

def pair_IoU(locations):
    """ compute IoU between each pair of boxes in
     the locations matrix
     [x_1i,y_1i,x_2i,y_2i]=locations[i,0:4]
    """
    M = locations.shape[0]
    Intersection = pair_Intersection(locations)
    Area = (locations[:,2] - locations[:,0] + 1 ) * (locations[:,3] - locations[:,1] + 1)
    Area_i = np.reshape(np.repeat(Area,M),(M,M))
    Area_j = np.reshape(np.tile(Area,M),(M,M))
    Union = Area_i + Area_j - Intersection
    if np.nonzero(Union ==0)[0].size:
        raise Exception("Union of boxes should not be zero")
    IoU = Intersection/Union
    return IoU


def find_local_argmax(Phi_labels, contributing_images, bbox_pred):
    """
    Find the index of the box with maximum score: [x1,y1,x2,y2]
    """
    M_cont = len(contributing_images)
    Phi_argmax = 4 * Phi_labels
    loc_argmax = bbox_pred[np.tile(contributing_images,4),np.hstack((Phi_argmax,Phi_argmax+1,Phi_argmax+2, Phi_argmax+3))]
    loc_argmax = np.reshape(loc_argmax,(M_cont,4),order='F')
    return loc_argmax


#!/usr/bin/env python

# --------------------------------------------------------
# LDDP
# Licensed under UC Berkeley's Standard Copyright [see LICENSE for details]
# Written by Samaneh Azadi
# --------------------------------------------------------

import numpy as np

from boxTools import *


class DPP():
    def dpp_greedy(self, S, scores_s, score_power, max_per_image, among_ims,
                   prob_threshold = 0.5, num_gt_per_img=1000, close_thr=0.0001):
        """
        Greedy optimization to select boxes
        S: similarity matrix
        scores_s : predicted scores over different categories

        """
        prob_thresh = prob_threshold
        S = S[among_ims,:][:,among_ims]
        scores_s = scores_s[among_ims]

        M = S.shape[0]

        #keep: selected_boxes
        keep = []

        #left : boxes not selected yet
        left = np.zeros((M,3))
        left[:,0] = np.arange(M) #box number
        left[:,1] = 1 # 0/1? Is the box left?
        selected_prob = []
        while (len(keep) < max_per_image) and sum(left[:,1])>0:
            z = np.zeros((M,1))
            z[keep] = 1
            sum_scores = (score_power*np.log(scores_s).T).dot(z)
            prob_rest = np.zeros((M,))
            left_indices = np.where(left[:,1]==1)[0]
            done_indices = np.where(left[:,1]==0)[0]
            if len(keep)>0:
                S_prev = S[keep,:][:,keep]
                det_D = np.linalg.det(S_prev)
                d_1 = np.linalg.inv(S_prev)
            else:
                det_D = 1
                d_1 = 0
            # ====================================================================
            #     |D  a^T|
            # det(|a    b|)= (b - a D^{-1} a^T)det(D)
            #
            # Here "D" = S_prev and "a","b" are the similarity values added by each single item
            # in left_indices.
            # To avoid using a for loop, we compute the above det for all items in left_indices
            # all at once through appropriate inner vector multiplications as the next line:

            # ====================================================================
            if len(keep)>0:
                prob_rest[left_indices] =- np.sum(np.multiply(np.dot(S[left_indices,:][:,keep],d_1),S[left_indices,:][:,keep]),1)

            prob_rest[left_indices] = np.log((prob_rest[left_indices] + S[left_indices,left_indices]) * det_D)+\
                           (sum_scores + score_power * np.log(scores_s[(left[left_indices,0]).astype(int)]))

            prob_rest[done_indices] = np.min(prob_rest)-100
            max_ind = np.argmax(prob_rest)
            close_inds = np.where(prob_rest >= (prob_rest[max_ind] + np.log(close_thr)))[0]
            tops_prob_rest = np.argsort(-prob_rest[close_inds]).astype(int)
            if len(keep) >= num_gt_per_img:
                break
            elif len(keep)> 0:
                cost = np.max(S[np.array(range(M))[close_inds][tops_prob_rest],:][:,keep],1)
                good_cost = list(np.where(cost <= prob_thresh)[0])

                if len(good_cost)>0:
                    ind = np.array(range(M))[close_inds][tops_prob_rest[good_cost[0]]]
                    keep.append(ind)
                    left[ind,1] = 0
                    selected_prob.append(prob_rest[max_ind])
                else:
                    left[:,1]=0

            else:
                keep.append(max_ind)
                left[max_ind,1] = 0
                selected_prob.append(prob_rest[max_ind])


        return keep,selected_prob


    def dpp_MAP(self, scores, boxes,sim_classes,score_thresh,epsilon,max_per_image,
                dpp_nms = 0.5, sim_power = 4, prob_threshold = 0.5,
                close_thr=0.00001):
       """
       DPP MAP inference
       """
       M0 = boxes.shape[0]
       num_classes = scores.shape[1]
       scores = scores[:,1:] #ignore background

       # consider only top 5 class scores per box
       num_ignored = scores.shape[1]-5
       sorted_scores = np.argsort(-scores,1)
       ignored_cols = np.reshape(sorted_scores[:,-num_ignored:],(M0*num_ignored))
       ignored_rows = np.repeat(range(0,sorted_scores.shape[0]),num_ignored)
       scores[ignored_rows,ignored_cols] = 0
       high_scores = np.nonzero(scores >= score_thresh)
       lbl_high_scores = high_scores[1]
       box_high_scores = high_scores[0]
       scores_s = np.reshape(scores[box_high_scores, lbl_high_scores],(lbl_high_scores.shape[0],))


       boxes = boxes[:,4:]
       boxes_s = np.reshape(boxes[np.tile(box_high_scores,4), np.hstack((4*lbl_high_scores,4*lbl_high_scores+1,\
       4*lbl_high_scores+2,4*lbl_high_scores+3))] ,(lbl_high_scores.shape[0],4),order='F')
       M = boxes_s.shape[0]
       sim_boxes = sim_classes[(lbl_high_scores),:][:,(lbl_high_scores)]
       sim_boxes = sim_boxes**sim_power
       keep_ = {}

       if M>0:
           IoU = pair_IoU(boxes_s)
           IoU[np.where(IoU<dpp_nms)] = 0
           # S = IoU * sim + \epsilon *I_M
           S = np.multiply(IoU,sim_boxes) + epsilon * np.eye(M,M)
           keep = self.dpp_greedy(S, scores_s, 1.0, max_per_image, np.array(range(M)), prob_threshold = prob_threshold,
                                  close_thr=close_thr)[0]
           keep_['box_id'] = box_high_scores[keep]
           keep_['box_cls'] = lbl_high_scores[keep]+1
       else:
          keep_['box_id'] = []
          keep_['box_cls'] = []

       return keep_

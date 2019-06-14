import torch
from torch import nn
import numpy as np
import cv2

# Heuristic choice based on that would scale a 16 pixel anchor up to 1000 pixels
BBOX_XFORM_CLIP = np.log(1000. / 16.)

EPSILON = 1e-7


def stack(x, dim=0, lib=np):
    if lib == np:
        return lib.stack(x, axis=dim)
    elif lib == torch:
        return lib.stack(x, dim=dim)
    else:
        raise NotImplementedError

def clamp(x, min=None, max=None, lib=np):
    if lib == np:
        return lib.clip(x, a_min=min, a_max=max)
    elif lib == torch:
        return lib.clamp(x, min=min, max=max)
    else:
        raise NotImplementedError

class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights=None, bbox_xform_clip=BBOX_XFORM_CLIP, lib=torch, relative_angle=True):
        """
        Arguments:
            weights (5-element tuple)  # None or xc,yc,w,h,theta
            bbox_xform_clip (float)
        """
        self.weights = weights
        if weights is not None:
            lenw = len(weights)
            assert 4 <= lenw <= 5
            if lenw == 4:
                self.weights = (weights[0],weights[1],weights[2],weights[3],1.0)
        else:
            self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.bbox_xform_clip = bbox_xform_clip
        if not (lib == np or lib == torch):
            raise NotImplementedError

        self.lib = lib
        self.relative_angle = relative_angle
        self.angle_multiplier = np.pi / 180  # deg to radians
        # self.angle_multiplier = 1. / 45

    def encode(self, unencode_boxes, reference_boxes):  # np.ones(5, dtype=np.float32)):
        '''
        :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
        :param reference_boxes: [H*W*num_anchors_per_location, 5]
        :return: encode_boxes [-1, 5]  # xc,yc,w,h,theta
        '''
        weights = self.weights
        lib = self.lib

        x_center, y_center, w, h, theta = \
            unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3], unencode_boxes[:, 4]
        reference_x_center, reference_y_center, reference_w, reference_h, reference_theta = \
            reference_boxes[:, 0], reference_boxes[:, 1], reference_boxes[:, 2], reference_boxes[:, 3], reference_boxes[
                                                                                                        :, 4]

        reference_w += EPSILON
        reference_h += EPSILON
        w += EPSILON
        h += EPSILON  # to avoid NaN in division and log below
        t_xcenter = (x_center - reference_x_center) / reference_w
        t_ycenter = (y_center - reference_y_center) / reference_h
        t_w = lib.log(w / reference_w)
        t_h = lib.log(h / reference_h)

        t_theta = (theta - reference_theta) if self.relative_angle else theta
        # t_theta[t_theta > 90] -= 90
        # t_theta[t_theta > 45] -= 90

        t_theta = t_theta * self.angle_multiplier  

        if weights is not None:
            wx, wy, ww, wh, wa = weights
            t_xcenter *= wx
            t_ycenter *= wy
            t_w *= ww
            t_h *= wh
            t_theta *= wa

        encode_boxes = stack([t_xcenter, t_ycenter, t_w, t_h, t_theta], dim=1, lib=lib)

        return encode_boxes

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        lib = self.lib

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]
        reference_theta = boxes[:, 4]

        wx, wy, ww, wh, wtheta = self.weights
        dx = rel_codes[:, 0::5] / wx
        dy = rel_codes[:, 1::5] / wy
        dw = rel_codes[:, 2::5] / ww
        dh = rel_codes[:, 3::5] / wh
        dtheta = rel_codes[:, 4::5] / wtheta

        # Prevent sending too large values into torch.exp()
        dw = clamp(dw, max=self.bbox_xform_clip, lib=lib)
        dh = clamp(dh, max=self.bbox_xform_clip, lib=lib)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = lib.exp(dw) * widths[:, None]
        pred_h = lib.exp(dh) * heights[:, None]
        pred_theta = dtheta / self.angle_multiplier
        if self.relative_angle:
             pred_theta += reference_theta[:, None]  # radians to degrees

        pred_boxes = lib.zeros_like(rel_codes)
        pred_boxes[:, 0::5] = pred_ctr_x
        pred_boxes[:, 1::5] = pred_ctr_y
        pred_boxes[:, 2::5] = pred_w
        pred_boxes[:, 3::5] = pred_h
        pred_boxes[:, 4::5] = pred_theta

        return pred_boxes

def convert_rect_to_pts2(anchors, lib=np):
    N = len(anchors)

    if lib == np:
        rect_pts = lib.zeros((N, 4, 2), dtype=lib.float32)
    elif lib == torch:
        rect_pts = lib.zeros((N, 4, 2), dtype=lib.float32, device=anchors.device)
    else:
        raise NotImplementedError

    if N == 0:
        return rect_pts

    cx = anchors[:,0]
    cy = anchors[:,1]
    w = anchors[:,2]
    h = anchors[:,3]
    angle = anchors[:,4] / 180. * np.pi

    b = lib.cos(angle)*0.5
    a = lib.sin(angle)*0.5

    rect_pts[:,0,0] = cx - a*h - b*w
    rect_pts[:,0,1] = cy + b*h - a*w
    rect_pts[:,1,0] = cx + a*h - b*w
    rect_pts[:,1,1] = cy - b*h - a*w
    rect_pts[:,2,0] = 2*cx - rect_pts[:,0,0]
    rect_pts[:,2,1] = 2*cy - rect_pts[:,0,1]
    rect_pts[:,3,0] = 2*cx - rect_pts[:,1,0]
    rect_pts[:,3,1] = 2*cy - rect_pts[:,1,1]

    return rect_pts

def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0

def area(int_pts, num_of_inter, lib=np):
    area = 0.0
    for i in range(num_of_inter - 2):
        area += lib.abs(trangle_area(int_pts, int_pts[2 * i + 2 : ], int_pts[2 * i + 4: ]))
    return area

def in_rect(pt_x, pt_y, pts, lib=np):
    ab = lib.zeros(2, dtype=lib.float32)
    ad = lib.zeros(2, dtype=lib.float32)
    ap = lib.zeros(2, dtype=lib.float32)

    ab[0] = pts[2] - pts[0];
    ab[1] = pts[3] - pts[1];

    ad[0] = pts[6] - pts[0];
    ad[1] = pts[7] - pts[1];

    ap[0] = pt_x - pts[0];
    ap[1] = pt_y - pts[1];

    abab = ab[0] * ab[0] + ab[1] * ab[1];
    abap = ab[0] * ap[0] + ab[1] * ap[1];
    adad = ad[0] * ad[0] + ad[1] * ad[1];
    adap = ad[0] * ap[0] + ad[1] * ap[1];

    return (abab >= abap) * (abap >= 0) * (adad >= adap) * (adap >= 0)

def inter2line(pts1, pts2, i, j, temp_pts, lib=np):
    a = lib.zeros(2, dtype=lib.float32)
    b = lib.zeros(2, dtype=lib.float32)
    c = lib.zeros(2, dtype=lib.float32)
    d = lib.zeros(2, dtype=lib.float32)

    a[0] = pts1[2 * i];
    a[1] = pts1[2 * i + 1];

    b[0] = pts1[2 * ((i + 1) % 4)];
    b[1] = pts1[2 * ((i + 1) % 4) + 1];

    c[0] = pts2[2 * j];
    c[1] = pts2[2 * j + 1];

    d[0] = pts2[2 * ((j + 1) % 4)];
    d[1] = pts2[2 * ((j + 1) % 4) + 1];

    area_abc = trangle_area(a, b, c);
    area_abd = trangle_area(a, b, d);

    if (area_abc * area_abd >= 0):
        return False;

    area_cda = trangle_area(c, d, a);
    area_cdb = area_cda + area_abc - area_abd;

    if (area_cda * area_cdb >= 0):
        return False;
    
    t = area_cda / (area_abd - area_abc);

    dx = t * (b[0] - a[0]);
    dy = t * (b[1] - a[1]);
    temp_pts[0] = a[0] + dx;
    temp_pts[1] = a[1] + dy;

    return True;

def reorder_pts(int_pts, num_of_inter, lib=np):

    if num_of_inter > 0:

        center = lib.zeros(2, dtype=lib.float32)
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i];
            center[1] += int_pts[2 * i + 1];
        center /= num_of_inter;

        vs = lib.zeros(16, dtype=lib.float32)
        v = lib.zeros(2, dtype=lib.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0];
            v[1] = int_pts[2 * i + 1] - center[1];
            d = lib.sqrt(v[0] * v[0] + v[1] * v[1]);
            v[0] = v[0] / d;
            v[1] = v[1] / d;
            if (v[1] < 0):
                v[0]= - 2 - v[0];
            
            vs[i] = v[0];

        for i in range(num_of_inter):
            if vs[i-1] > vs[i]:
                temp = vs[i].clone() if lib == torch else vs[i]
                tx = int_pts[2*i].clone() if lib == torch else int_pts[2*i]
                ty = int_pts[2*i+1].clone() if lib == torch else int_pts[2*i+1]
                j=i;
                while (j > 0 and vs[j-1] > temp):
                    vs[j] = vs[j-1];
                    int_pts[j*2] = int_pts[j*2-2];
                    int_pts[j*2+1] = int_pts[j*2-1];
                    j -= 1;
              
                vs[j] = temp;
                int_pts[j*2] = tx;
                int_pts[j*2+1] = ty;

def inter_pts(pts1, pts2, lib=np):

    int_pts = lib.zeros(16, dtype=lib.float32)

    num_of_inter = 0
    for i in range(4):
        if in_rect(pts1[2 * i], pts1[2 * i + 1], pts2, lib=lib):
            int_pts[num_of_inter * 2] = pts1[2 * i];
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
            num_of_inter += 1
        
        if in_rect(pts2[2 * i], pts2[2 * i + 1], pts1, lib=lib):
            int_pts[num_of_inter * 2] = pts2[2 * i];
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
            num_of_inter += 1

    for i in range(4):
        for j in range(4):
            temp_pts = lib.zeros(2, dtype=lib.float32)
            has_pts = inter2line(pts1, pts2, i, j, temp_pts, lib=lib)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0];
                int_pts[num_of_inter * 2 + 1] = temp_pts[1];
                num_of_inter += 1

    return num_of_inter, int_pts

def iou_rotate_cpu2(boxes1, boxes2, lib=np):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    N = len(boxes1)
    assert N == len(boxes2)
    ious = lib.zeros(N, dtype=lib.float32)

    box1_pts = convert_rect_to_pts2(boxes1, lib=lib)
    box2_pts = convert_rect_to_pts2(boxes2, lib=lib)

    for i, box1 in enumerate(boxes1):
        box2 = boxes2[i]

        pts1 = box1_pts[i].flatten()
        pts2 = box2_pts[i].flatten()

        num_of_inter, int_pts = inter_pts(pts1, pts2, lib=lib)
        iou = 0.0
        if num_of_inter > 0:
            reorder_pts(int_pts, num_of_inter, lib=lib)
            int_area = area(int_pts, num_of_inter, lib=lib)

            iou = int_area * 1.0 / (area1[i] + area2[i] - int_area)
        ious[i] = iou

    return ious

def iou_rotate_cpu(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    assert len(boxes1) == len(boxes2)
    ious = []
    for i, box1 in enumerate(boxes1):
        box2 = boxes2[i]
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

        int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
        iou = 0.0
        if int_pts is not None:
            order_pts = cv2.convexHull(int_pts, returnPoints=True)

            int_area = cv2.contourArea(order_pts)

            iou = int_area * 1.0 / (area1[i] + area2[i] - int_area)
        ious.append(iou)

    return np.array(ious, dtype=np.float32)

def compute_iou_loss(pred, target, base_box, box_coder):
    pred_box = box_coder.decode(pred, base_box)
    gt_box = box_coder.decode(target, base_box)

    ious = iou_rotate_cpu2(pred_box, gt_box, lib=torch)
    iou_loss = -torch.log(ious)
    return iou_loss.mean()

if __name__ == '__main__':
    from giou_loss import smooth_l1_loss, SimpleLinearModel

    def normalize_rect_tensors(rects):
        r = torch.zeros_like(rects)
        r[:,:4] = (rects[:,:4] - 100) / 200.0
        r[:,-1] = rects[:,-1] / 180. * np.pi
        return r

    anchor = np.array([
        [80,80,160,160, 0],  # xc yc w h angle
        [90,90,105,105, 30],
        [50,50,50,50, -30],
    ], dtype=np.float32)
    bbox_gt = np.array([
        [90,90,170,170, 0], 
        [100,100,95,95, 40],
        [55,55,70,70, -10],
    ], dtype=np.float32)

    N = len(bbox_gt)

    ious = iou_rotate_cpu(bbox_gt, anchor)
    ious2 = iou_rotate_cpu2(bbox_gt, anchor)
    print(ious)
    print(ious2)

    t_bbox_gt = torch.from_numpy(bbox_gt)
    t_anchor = torch.from_numpy(anchor)

    box_coder = BoxCoder(weights=(1.0,1.0,1.0,1.0,1.0))
    regression_targets = box_coder.encode(
        t_bbox_gt, t_anchor
    )

    model = SimpleLinearModel(regression_cn=5)

    import torch.optim as optim
    lr = 1e-2
    n_iters = 100

    optimizer = optim.Adam(model.parameters(), lr=lr)#, betas=(0.9, 0.999))
    for n in range(n_iters):
        
        regression_pred = model(normalize_rect_tensors(t_anchor), normalize_rect_tensors(t_bbox_gt))

        l1_loss = smooth_l1_loss(regression_pred, regression_targets, beta=1.0/9)
        iou_loss = compute_iou_loss(regression_pred, regression_targets, t_anchor, box_coder)

        loss = iou_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bbox_pred = box_coder.decode(
                regression_pred.view(-1, 5), t_anchor.view(-1, 5)
            ).numpy()
            mean_iou = np.mean(iou_rotate_cpu(bbox_pred, bbox_gt))
            print("Iter %d) Loss: %.3f, Mean IoU: %.2f"%(n, loss.item(), mean_iou))

    # anchor_pts = convert_rect_to_pts2(t_anchor, lib=torch)
    # gt_pts = convert_rect_to_pts2(t_bbox_gt, lib=torch)

    # pts1 = anchor_pts[0].flatten()
    # pts2 = gt_pts[0].flatten()
    # area1 = (t_anchor[:, 2] * t_anchor[:, 3])[0]
    # area2 = (t_bbox_gt[:, 2] * t_bbox_gt[:, 3])[0]

    # num_of_inter, int_pts = inter_pts(pts1, pts2, lib=torch)
    # int_pts_numpy = int_pts.detach().numpy().copy()

    # if num_of_inter > 0:
    #     reorder_pts(int_pts, num_of_inter, lib=torch)
    #     int_area = area(int_pts, num_of_inter, lib=torch)
    #     iou = int_area * 1.0 / (area1 + area2 - int_area)

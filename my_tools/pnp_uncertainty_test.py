import cv2
import numpy as np
import numpy.random as npr
import open3d

from conv_vertex_field import solve_pnp, project_points, get_voting_mean_pixel, draw_voting_results

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

if __name__ == '__main__':

    point_keypoints = np.array([
        [0.04965743, 0.0696693,  0.00412764],
        [-0.04912861,  0.06956633, -0.0067097 ],
        [-0.04532503, -0.06941872, -0.02012832],
        [ 0.05041973, -0.069449,    0.00409536],
        [ 0.01400065, -0.06746875, -0.04612001],
        [-0.01939844, -0.06946825,  0.0465417 ],
        [ 0.00469201,  0.06574368, -0.05085601],
        [-0.00595804,  0.0699745,   0.04865649],
        [0,0,0]
    ])
    point_cloud = open3d.read_point_cloud("/home/bot/hd/datasets/FAT/models/002_master_chef_can/002_master_chef_can.pcd")
    points = np.asarray(point_cloud.points)

    intrinsic_matrix = np.array([
        [432.1, 0.0, 480.0], 
        [0.0, 432.1, 270.0], 
        [0.0, 0.0, 1.0]
    ])

    img_file = "/home/bot/Documents/practice/render/out_test/7_img.jpg"
    img = cv2.imread(img_file)

    pose = np.array([
         [ 0.0158, -0.9995,  0.0288, -0.632 ],
         [ 0.9373,  0.0248,  0.3477,  0.2695],
         [ 0.3482, -0.0215, -0.9372, -0.6967],
         [ 0.,      0.    ,  0.    ,  1.    ]
    ])

    full_img2 = img.copy()
    project_points(points, pose[:3,:3], pose[:3,3], intrinsic_matrix, img=full_img2)


    keypoints_2d = project_points(point_keypoints, pose[:3,:3], pose[:3,3], intrinsic_matrix, img=None)
    print(keypoints_2d)

    x1, y1 = (801,62)
    ori_h, ori_w = (78, 152)
    x2 = x1 + ori_w + 1
    y2 = y1 + ori_h + 1
    bbox = np.array([x1,y1,x2,y2])
    H, W = (96, 96)

    # now get prediction results
    results_pred = np.load("results_pred.npy")
    C = len(results_pred)
    assert C == len(point_keypoints)

    # visualize pred results
    cropped_img = img[y1:y2,x1:x2]
    cropped_img = cv2.resize(cropped_img, (W,H))
    canvas_heatmap_pred = np.zeros((C * H, W*3, 3), dtype=np.uint8)
    for jx in range(C):
        sy1, sy2 = jx * W, (jx+1) * W
        mean_pred_px, heatmap_pred = draw_voting_results(cropped_img, results_pred[jx], top_k=100)

        cropped_img_copy = cropped_img.copy()
        kp = keypoints_2d[jx] - bbox[:2]
        kp_px = np.array([kp[0] / ori_w * W, kp[1] / ori_h * H]).astype(np.int32)
        cv2.circle(cropped_img_copy, tuple(kp_px), 2, GREEN, -1)

        canvas_heatmap_pred[sy1:sy2, :W] = heatmap_pred
        canvas_heatmap_pred[sy1:sy2, W:W*2] = mean_pred_px
        canvas_heatmap_pred[sy1:sy2, W*2:] = cropped_img_copy


    # get projected keypoints and run PnP
    proj_pred_keypoints = np.array([get_voting_mean_pixel(results_pred[jx], top_k=20) for jx in range(C)])
    # set 2d values back to original image 
    proj_pred_keypoints = proj_pred_keypoints * np.array([ori_w / W, ori_h / H]) + np.array([x1, y1])

    # solve pnp n draw 2d projected pts 
    pnp_algorithm = cv2.SOLVEPNP_EPNP # cv2.SOLVEPNP_ITERATIVE

    D = proj_pred_keypoints.reshape((C,1,2))

    # # compute statistics of result
    top_k = 100
    mean_var = np.zeros(C)
    for jx in range(C):
        result = results_pred[jx]
        top_result = result[:top_k]
        top_occurrences = top_result[:,-1]
        top_pixels = top_result[:,:2]

        w_cov_x = np.cov(top_pixels[:,0], fweights=top_occurrences)
        w_cov_y = np.cov(top_pixels[:,1], fweights=top_occurrences)
        print("var x: %.3f, var y: %.3f"%(w_cov_x, w_cov_y))

        mean_var[jx] = (w_cov_y + w_cov_x) / 2

    ranks = np.argsort(mean_var)
    print(ranks)

    pick = 6
    kp_ranked = point_keypoints[ranks][:pick]#.copy()
    D_ranked = D[ranks][:pick]#.copy()

    success, M = solve_pnp(kp_ranked, D_ranked, intrinsic_matrix, flags=pnp_algorithm)
    img2 = img.copy()
    project_points(points, M[:3,:3], M[:3,3], intrinsic_matrix, img=img2)
    print(M)

    cv2.imshow("img", img)
    cv2.imshow("gt_proj_2d", full_img2)    
    cv2.imshow("pred_heat, mean pred, gt", canvas_heatmap_pred)
    cv2.imshow("pred_proj_2d_vote_pnp", img2)
    cv2.waitKey(0)


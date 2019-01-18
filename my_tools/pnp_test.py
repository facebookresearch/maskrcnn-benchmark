"""
https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
"""
import cv2
import numpy as np
import open3d

from conv_test import create_cloud, backproject_camera
from conv_vertex_field import draw_axis_pose
from transforms3d.quaternions import quat2mat#, mat2quat

import numpy.linalg as la

def get_vector_angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def draw_cuboid_2d(img2, cuboid, color):
    assert len(cuboid) == 8
    points = [tuple(pt) for pt in cuboid]
    for ix in range(len(points)):
        pt = points[ix]
        cv2.putText(img2, "%d"%(ix), pt, cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
        cv2.circle(img2, pt, 3, (0,255,0), -1)

    lines = [[0,1],[0,2],[1,3],[2,3],
         [4,5],[4,6],[5,7],[6,7],
         [0,4],[1,5],[2,6],[3,7]]

    for line in lines:
        pt1 = points[line[0]]
        pt2 = points[line[1]]
        cv2.line(img2, pt1, pt2, color)

def get_data():
    # im_file = "./datasets/FAT/data/mixed/temple_0/000295.right.jpg"
    # # points indexed 3,4 are the front and back ones
    # model_points = np.array([
    #     [-0.05080691, -0.041769,   -0.02880055],
    #     [-0.05080691,  0.04176 ,   -0.02880055],
    #     [-0.05080691, -0.041769,    0.02880038],
    #     [-0.05080691,  0.04176 ,    0.02880038],
    #     [ 0.0508229 , -0.041769,   -0.02880055],
    #     [ 0.0508229 ,  0.04176 ,   -0.02880055],
    #     [ 0.0508229 , -0.041769,    0.02880038],
    #     [ 0.0508229 ,  0.04176 ,    0.02880038],
    #     [ 0,           0,           0         ] # add centroid
    # ])
    # image_points = np.array([
    #     [569, 299],
    #     [526, 309],
    #     [589, 329],
    #     [546, 337],
    #     [538, 348],
    #     [495, 355],
    #     [561, 379],
    #     [517, 384],
    # ], dtype=np.float32)
    # # image_points = np.array([
    # #     [575, 304],
    # #     [530, 311],
    # #     [580, 332],
    # #     [540, 332],
    # #     [533, 350],
    # #     [496, 350],
    # #     [564, 383],
    # #     [514, 389],
    # # ], dtype=np.float32)
    # cls = 9

    # im_file = "./datasets/FAT/data/mixed/temple_0/001600.right.jpg"
    # model_points = np.array([
    #     [-0.10122322, -0.04362476, -0.00781424],
    #     [-0.10122322,  0.04369109, -0.00781424],
    #     [-0.10122322, -0.04362476,  0.00778602],
    #     [-0.10122322,  0.04369109,  0.00778602],
    #     [ 0.10129841, -0.04362476, -0.00781424],
    #     [ 0.10129841,  0.04369109, -0.00781424],
    #     [ 0.10129841, -0.04362476,  0.00778602],
    #     [ 0.10129841,  0.04369109,  0.00778602],
    # ])
    # image_points = np.array([
    #     [400, 457],
    #     [452, 446],
    #     [397, 445],
    #     [449, 435],
    #     [492, 414],
    #     [547, 404],
    #     [488, 401],
    #     [543, 392],
    # ], dtype=np.float32)
    # cls = 17

    im_file = "./datasets/FAT/data/mixed/temple_0/000127.left.jpg"
    model_points = np.array([
        [-0.081932, -0.106709, -0.035852],
        [-0.081932,  0.106668, -0.035852],
        [-0.081932, -0.106709,  0.035837],
        [-0.081932,  0.106668,  0.035837],
        [ 0.081727, -0.106709, -0.035852],
        [ 0.081727,  0.106668, -0.035852],
        [ 0.081727, -0.106709,  0.035837],
        [ 0.081727,  0.106668,  0.035837]
    ])
    image_points = [[471, 395],
        [543, 477],
        [471, 355],
        [545, 432],
        [369, 424],
        [428, 517],
        [366, 382],
        [426, 470]]
    cls = 2

    # im_file = "./datasets/FAT/data/mixed/temple_0/001600.right.jpg"
    # model_points = [
    #     [-0.05080691, -0.041769 ,  -0.02880055],
    #     [-0.05080691,  0.04176  ,  -0.02880055],
    #     [-0.05080691, -0.041769 ,   0.02880038],
    #     [-0.05080691,  0.04176  ,   0.02880038],
    #     [ 0.0508229 , -0.041769 ,  -0.02880055],
    #     [ 0.0508229 ,  0.04176  ,  -0.02880055],
    #     [ 0.0508229 , -0.041769 ,   0.02880038],
    #     [ 0.0508229 ,  0.04176  ,   0.02880038]
    #  ]
    # image_points = [[684, 341],
    #  [744, 323],
    #  [683, 345],
    #  [746, 325],
    #  [709, 411],
    #  [769, 394],
    #  [709, 419],
    #  [774, 401]]
    # cls = 9

    # im_file = "./datasets/FAT/data/mixed/temple_0/000127.left.jpg"
    # model_points = [[-0.03359818, -0.050927,   -0.0338576 ],
    #  [-0.03359818,  0.050916 ,  -0.0338576 ],
    #  [-0.03359818, -0.050927 ,   0.03385666],
    #  [-0.03359818,  0.050916 ,   0.03385666],
    #  [ 0.03361415, -0.050927 ,  -0.0338576 ],
    #  [ 0.03361415,  0.050916 ,  -0.0338576 ],
    #  [ 0.03361415, -0.050927 ,   0.03385666],
    #  [ 0.03361415,  0.050916 ,   0.03385666]]
    # image_points = [[730, 411],
    #  [722, 476],
    #  [699, 434],
    #  [691, 500],
    #  [685, 389],
    #  [679, 452],
    #  [653, 410],
    #  [648, 474]]

    # cls = 4

    model_points = np.vstack((model_points, [0,0,0]))
    center = np.mean(image_points, axis=0)
    image_points = np.vstack((image_points, center)).astype(np.float32)

    return model_points, image_points, im_file, cls

from transforms3d.euler import euler2mat

def get_rotate_mat(angle, axis=0):
    rad = np.radians(angle)
    axes = [0,0,0]
    axes[axis] = rad
    return euler2mat(*axes)

def get_rotate(R_array, init_R=np.identity(3)):
    RR = init_R.copy()
    for R in R_array:
        RR = np.dot(R, RR)
    return RR


if __name__ == '__main__':

    points_file = "./datasets/FAT/points_all_orig.npy"
    points = np.load(points_file)

    # Read Image
    model_points, image_points, im_file, cls = get_data()
    depth_file = im_file.replace(".jpg",".depth.png")


    intrinsics = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])
    factor_depth=10000

    img = cv2.imread(im_file)
    # img = np.zeros((800,800,3), dtype=np.uint8) #cv2.imread("headPose.jpg")
    H,W,_ = img.shape

    # load depth and cloud
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

    rgb = img.copy()[:,:,::-1]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    X = backproject_camera(depth, intrinsics, factor_depth)
    scene_cloud = create_cloud(X.T, colors=rgb.reshape((H*W,3)))

    ordering = np.array([0,1,2,3,4,5,6,7,8])
    # ordering = np.array([6,7,4,5,2,3,0,1,8])  # rotate 180 by one axis
    # ordering = np.array([4,5,0,1,6,7,2,3,8])  # rotate 90 by one axis
    # ordering = np.array([3,2,1,0,7,6,5,4,8])  # rotate 180 by other axis
    # ordering = np.array([5,4,7,6,1,0,3,2,8])  # rotate 180 by other axis

    image_points = image_points[ordering]

    camera_matrix = intrinsics

    print("Camera Matrix :\n %s"%(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    pnp_algorithm = cv2.SOLVEPNP_ITERATIVE 
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=pnp_algorithm)
    # success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, flags=pnp_algorithm)#, iterationsCount=1000)
     
    # print("Rotation Vector:\n %s"%(rotation_vector))
    # print("Translation Vector:\n %s"%(translation_vector))

    R, j = cv2.Rodrigues(rotation_vector)
    im = img.copy()
    im = draw_axis_pose(im, R, translation_vector.squeeze(), camera_matrix, dist_coeffs)
     
    draw_cuboid_2d(im, image_points[:8], (255,0,0)) 
    cv2.circle(im, tuple(image_points[-1]), 4, (0,0,255), -1)

    # AAA
    theeight = np.array([
        [1,1,1],  # top right, back
        [1,-1,1],  # bottom right, back
        [1,1,-1],  # top right, front
        [1,-1,-1],  # bottom right, front
        
        [-1,1,1],  # top left, back
        [-1,-1,1],  # bottom left, back
        [-1,1,-1],  # top left, front
        [-1,-1,-1],  # bottom left, front

        [0,0,0]  # centroid
    ], dtype=np.float32)
    theeight_ordering = np.array([0,1,2,3,4,5,6,7,8])
    theeight_points = image_points[theeight_ordering] 
    side_mapping = [[0,4],[0,1],[0,2]] # x (left-right), y (bottom-top), z (front-back)
    sides = np.zeros((3), dtype=np.float32)
    for ix,m in enumerate(side_mapping):
        dist = np.linalg.norm(theeight_points[m[0]] - theeight_points[m[1]]) 
        sides[ix] = dist / 2
    theeight_model_points = theeight * sides
    s,rvec2,t = cv2.solvePnP(theeight_model_points, theeight_points, camera_matrix, dist_coeffs)
    R2, j = cv2.Rodrigues(rvec2)

    # thesix = np.array([
    #     [1,1,1],  # top right
    #     [1,-1,1],  # mid right
    #     [1,-1,-1],  # bottom right
    #     [-1,-1,-1],  # bottom left
    #     [-1,1,-1],  # mid left
    #     [-1,1,1],  # top left
    #     [0,0,0]  # centroid
    # ], dtype=np.float32)
    # thesix_ordering = np.array([0,2,6,7,5,1,8])
    # # thesix_ordering = np.array([0,2,6,7,5,1,8])
    # thesix_points = image_points[thesix_ordering] 

    # sides = np.zeros((3), dtype=np.float32)
    # for ix,o in enumerate(thesix_ordering[1:4]):
    #     p1i = thesix_ordering[ix]
    #     p2i = thesix_ordering[ix+1]
    #     dist = np.linalg.norm(image_points[p1i] - image_points[p2i]) 
    #     sides[ix] = dist / 2
    # sides = sides[[2,0,1]]
    # thesix_model_points = thesix * sides
    # # thesix_points = image_points[thesix_ordering]
    # s,rvec2,t = cv2.solvePnP(thesix_model_points, thesix_points, camera_matrix, dist_coeffs)
    # R2, j = cv2.Rodrigues(rvec2)
    # """
    # red direction: 2,3
    # blue direction: 3,4,5
    # green direction: 1,2,3

    # Need to change to:
    # red direction: 1,2,3
    # blue direction: 2,3
    # green direction: 3,4,5
    # """
    # for ix, pt in enumerate(thesix_points[:-1]):
    #     pt = tuple(pt)
    #     cv2.circle(im2, pt, 4, (0,255,0), -1)
    #     cv2.putText(im2, "%d"%(ix), pt, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255)) 
    #     cv2.line(im2, pt, tuple(thesix_points[(ix+1)%6]), (0,0,255))

    im2 = img.copy()
    im2 = draw_axis_pose(im2, R2, translation_vector.squeeze(), camera_matrix, dist_coeffs)

    # Display image
    cv2.imshow("GT", im)
    cv2.imshow("P", im2)
    cv2.waitKey(0)

    object_points = points[cls]
    M = np.identity(4)
    M[:3,:3] = R
    M[:3,-1] = translation_vector.squeeze()
    object_cloud = create_cloud(object_points, T=M)

    open3d.draw_geometries([scene_cloud, object_cloud])


    Rc = R.T
    R2c = R2.T
    angle_matrix = np.zeros((3,3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            angle_matrix[i,j] = get_vector_angle( Rc[i], R2c[j] )
    similarity_matrix = np.sin(angle_matrix)

    best = np.argsort(similarity_matrix.flatten()) # small to large  (smaller means more similar)
    assigned = 0
    axes = np.zeros((3), dtype=np.int32)
    last_axis_sign = 1
    for ix,b in enumerate(best):
        i = b // 3 
        j = b % 3
        # check if 
        if axes[i] != 0:
            continue
        # get the sign (+/-) of cosine similarity
        cos_similarity = np.cos(angle_matrix[i,j])
        sign = np.sign(cos_similarity)
        axes[i] = sign * (j+1)
        # last_axis_sign *= sign
        assigned += 1
        if assigned == 2: # np.sum(axes != 0) >= 2: # only need to know direction of 2 axes to know the third
            # axes[axes==0] = last_axis_sign * ((1+2+3) - np.abs(axes).sum())
            break

    # Suppose axis 1,2,3 is x,y,z respectively (LHR)
    aa = {
        (1,2,3): [0,1,2,3,4,5,6,7], # default
        (1,-2,-3): [3,2,1,0,7,6,5,4],  # rotated 180 around x
        (1,3,-2): [1,3,0,2,5,7,4,6],  # rotated 90 clockwise around x
        (1,-3,2): [2,0,3,1,6,4,7,5],   # rotated 270 clockwise around x

        # start with -1 (rotate 180 around y from default)
        (-1,2,-3): [6,7,4,5,2,3,0,1],   # 
        (-1,-2,3): [5,4,7,6,1,0,3,2],   # then rotate 180 around x
        (-1,3,2): [7,5,6,4,3,1,2,0],   # then rotate 90 clockwise around x
        (-1,-3,-2): [4,6,5,7,0,2,1,3],   # then rotate 270 clockwise around x

        # start with 2 (rotate 270 clockwise around z from default)
        (2,-1,3): [1,5,3,7,0,4,2,6],   # 
        (2,1,-3): [7,3,5,1,6,2,4,0],   # then rotate 180 around y
        (2,3,1): [],   # then rotate 90 clockwise around y
        (2,-3,-1): [],   # then rotate 270 clockwise around y

        # start with -2 (rotate 90 clockwise around z from default)
        (-2,1,3): [],   # 
        (-2,-1,-3): [],   # then rotate 180 around y
        (-2,3,-1): [],   # then rotate 90 clockwise around y
        (-2,-3,1): [],   # then rotate 270 clockwise around y

        # start with 3 (rotate 90 clockwise around y from default)
        (3,2,-1): [],   # 
        (3,-2,1): [],   # then rotate 180 around z
        (3,-1,-2): [],   # then rotate 90 clockwise around z
        (3,1,2): [],   # then rotate 270 clockwise around z

        # start with -3 (rotate 270 clockwise around y from default)
        (-3,2,1): [],   # 
        (-3,-2,-1): [],   # then rotate 180 around z
        (-3,-1,2): [],   # then rotate 90 clockwise around z
        (-3,1,-2): [],   # then rotate 270 clockwise around z
    }

    aa = {
        (1,2,3): [get_rotate_mat(0,0)], # default
        (1,-2,-3): [get_rotate_mat(180,0)],  # rotated 180 around x
        (1,3,-2): [get_rotate_mat(90,0)],  # rotated 90 clockwise around x
        (1,-3,2): [get_rotate_mat(270,0)],   # rotated 270 clockwise around x

        # start with -1 (rotate 180 around y from default)
        (-1,2,-3): [get_rotate_mat(180,1)],   # 
        (-1,-2,3): [get_rotate_mat(180,1), get_rotate_mat(180,0)],   # then rotate 180 around x
        (-1,3,2): [get_rotate_mat(180,1), get_rotate_mat(90,0)],   # then rotate 90 clockwise around x
        (-1,-3,-2): [get_rotate_mat(180,1), get_rotate_mat(270,0)],   # then rotate 270 clockwise around x

        # start with 2 (rotate 270 clockwise around z from default)
        (2,-1,3): [1,5,3,7,0,4,2,6],   # 
        (2,1,-3): [7,3,5,1,6,2,4,0],   # then rotate 180 around y
        (2,3,1): [],   # then rotate 90 clockwise around y
        (2,-3,-1): [],   # then rotate 270 clockwise around y

        # start with -2 (rotate 90 clockwise around z from default)
        (-2,1,3): [],   # 
        (-2,-1,-3): [],   # then rotate 180 around y
        (-2,3,-1): [],   # then rotate 90 clockwise around y
        (-2,-3,1): [],   # then rotate 270 clockwise around y

        # start with 3 (rotate 90 clockwise around y from default)
        (3,2,-1): [],   # 
        (3,-2,1): [],   # then rotate 180 around z
        (3,-1,-2): [],   # then rotate 90 clockwise around z
        (3,1,2): [],   # then rotate 270 clockwise around z

        # start with -3 (rotate 270 clockwise around y from default)
        (-3,2,1): [],   # 
        (-3,-2,-1): [],   # then rotate 180 around z
        (-3,-1,2): [],   # then rotate 90 clockwise around z
        (-3,1,-2): [],   # then rotate 270 clockwise around z
    }

    m = np.zeros(im.shape, dtype=np.uint8)
    R45 = get_rotate([get_rotate_mat(45,0), get_rotate_mat(30,1)])
    imgpts, _ = cv2.projectPoints(theeight * 0.1, R45, np.array([0,0,1.0]), camera_matrix, dist_coeffs)
    # for ix, pt in enumerate(imgpts.squeeze()):
    #     pt = tuple(pt)
    #     cv2.circle(m, pt, 4, (0,0,255), -1)
    #     cv2.putText(m, "%d"%(ix), pt, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
    draw_cuboid_2d(m, imgpts.squeeze()[:-1], (255,0,0))
    cv2.imshow("m", m)
    # cv2.waitKey(0)

    m = np.zeros(im.shape, dtype=np.uint8)
    imgpts, _ = cv2.projectPoints(theeight * 0.1, get_rotate([get_rotate_mat(180,0)], R45), np.array([0,0,1.0]), camera_matrix, dist_coeffs)
    draw_cuboid_2d(m, imgpts.squeeze()[:-1], (0,0,255))
    # for ix, pt in enumerate(imgpts.squeeze()):
    #     pt = tuple(pt)
    #     cv2.circle(m, pt, 4, (0,0,255), -1)
    #     cv2.putText(m, "%d"%(ix), pt, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))

    cv2.imshow("m2", m)
    cv2.waitKey(0)

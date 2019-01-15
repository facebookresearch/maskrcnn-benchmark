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

    # im_file = "./datasets/FAT/data/mixed/temple_0/000127.left.jpg"
    # model_points = np.array([
    #     [-0.081932, -0.106709, -0.035852],
    #     [-0.081932,  0.106668, -0.035852],
    #     [-0.081932, -0.106709,  0.035837],
    #     [-0.081932,  0.106668,  0.035837],
    #     [ 0.081727, -0.106709, -0.035852],
    #     [ 0.081727,  0.106668, -0.035852],
    #     [ 0.081727, -0.106709,  0.035837],
    #     [ 0.081727,  0.106668,  0.035837]
    # ])
    # image_points = [[471, 395],
    #     [543, 477],
    #     [471, 355],
    #     [545, 432],
    #     [369, 424],
    #     [428, 517],
    #     [366, 382],
    #     [426, 470]]
    # cls = 2

    im_file = "./datasets/FAT/data/mixed/temple_0/001600.right.jpg"
    model_points = [
        [-0.05080691, -0.041769 ,  -0.02880055],
        [-0.05080691,  0.04176  ,  -0.02880055],
        [-0.05080691, -0.041769 ,   0.02880038],
        [-0.05080691,  0.04176  ,   0.02880038],
        [ 0.0508229 , -0.041769 ,  -0.02880055],
        [ 0.0508229 ,  0.04176  ,  -0.02880055],
        [ 0.0508229 , -0.041769 ,   0.02880038],
        [ 0.0508229 ,  0.04176  ,   0.02880038]
     ]
    image_points = [[684, 341],
     [744, 323],
     [683, 345],
     [746, 325],
     [709, 411],
     [769, 394],
     [709, 419],
     [774, 401]]
    cls = 9

    im_file = "./datasets/FAT/data/mixed/temple_0/000127.left.jpg"
    model_points = [[-0.03359818, -0.050927,   -0.0338576 ],
     [-0.03359818,  0.050916 ,  -0.0338576 ],
     [-0.03359818, -0.050927 ,   0.03385666],
     [-0.03359818,  0.050916 ,   0.03385666],
     [ 0.03361415, -0.050927 ,  -0.0338576 ],
     [ 0.03361415,  0.050916 ,  -0.0338576 ],
     [ 0.03361415, -0.050927 ,   0.03385666],
     [ 0.03361415,  0.050916 ,   0.03385666]]
    image_points = [[730, 411],
     [722, 476],
     [699, 434],
     [691, 500],
     [685, 389],
     [679, 452],
     [653, 410],
     [648, 474]]

    cls = 4

    model_points = np.vstack((model_points, [0,0,0]))
    center = np.mean(image_points, axis=0)
    image_points = np.vstack((image_points, center)).astype(np.float32)

    return model_points, image_points, im_file, cls

points_file = "./datasets/FAT/points_all_orig.npy"
points = np.load(points_file)

# Read Image
model_points, image_points, im_file, cls = get_data()
depth_file = im_file.replace(".jpg",".depth.png")


intrinsics = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])
factor_depth=10000

im = cv2.imread(im_file)
# im = np.zeros((800,800,3), dtype=np.uint8) #cv2.imread("headPose.jpg")
H,W,_ = im.shape

# load depth and cloud
depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

rgb = im.copy()[:,:,::-1]
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

thesix = np.array([
    [1,1,1],  # top right
    [1,-1,1],  # mid right
    [1,-1,-1],  # bottom right
    [-1,-1,-1],  # bottom left
    [-1,1,-1],  # mid left
    [-1,1,1],  # top left
    [0,0,0]  # centroid
], dtype=np.float32)
thesix_ordering = np.array([0,2,6,7,5,1,8])
# thesix_ordering = np.array([0,2,6,7,5,1,8])
thesix_points = image_points[thesix_ordering] 

camera_matrix = intrinsics

print("Camera Matrix :\n %s"%(camera_matrix))
 
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
pnp_algorithm = cv2.SOLVEPNP_ITERATIVE 
success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=pnp_algorithm)
# success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, flags=pnp_algorithm)#, iterationsCount=1000)
 
print("Rotation Vector:\n %s"%(rotation_vector))
print("Translation Vector:\n %s"%(translation_vector))
 
# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose

R, j = cv2.Rodrigues(rotation_vector)
# print(R)
# (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.1)]), R, translation_vector, camera_matrix, dist_coeffs)
 
draw_cuboid_2d(im, image_points[:8], (255,0,0)) 
cv2.circle(im, tuple(image_points[-1]), 4, (0,0,255), -1)
# _,rvec2,_ = cv2.solvePnP(thesix, thesix_points, camera_matrix, dist_coeffs, flags=pnp_algorithm)
# s,rvec2,t = cv2.solvePnP(thesix[[4,5,0,-1]], image_points[[2,6,7,-1]], camera_matrix, dist_coeffs)

sides = np.zeros((3), dtype=np.float32)
for ix,o in enumerate(thesix_ordering[1:4]):
    p1i = thesix_ordering[ix]
    p2i = thesix_ordering[ix+1]
    dist = np.linalg.norm(image_points[p1i] - image_points[p2i]) 
    sides[ix] = dist / 2
sides = sides[[2,0,1]]
thesix_model_points = thesix * sides
thesix_points = image_points[thesix_ordering]
s,rvec2,t = cv2.solvePnP(thesix_model_points, thesix_points, camera_matrix, dist_coeffs)
R2, j = cv2.Rodrigues(rvec2)
im = draw_axis_pose(im, R2, translation_vector.squeeze(), camera_matrix, dist_coeffs)

 
# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)

object_points = points[cls]
M = np.identity(4)
M[:3,:3] = R
M[:3,-1] = translation_vector.squeeze()
object_cloud = create_cloud(object_points, T=M)

open3d.draw_geometries([scene_cloud, object_cloud])


similarity_matrix = np.zeros((3,3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        similarity_matrix[i,j] = np.sin(get_vector_angle( R.T[i], R2.T[j] ))

best = np.argsort(similarity_matrix.flatten())
best // 3 


"""
Module for cv2 utility functions and maintaining version compatibility
between 3.x and 4.x
"""
import cv2


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    ver = cv2.__version__
    if ver.startswith('4') or ver.startswith('2'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif ver.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 2, 3 or 4 to call this method')

    return contours, hierarchy

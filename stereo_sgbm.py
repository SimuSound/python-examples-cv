#####################################################################

# Example : stereo vision from 2 connected cameras using Semi-Global
# Block Matching. For usage: python3 ./stereo_sgbm.py -h

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015-2020 Engineering & Computer Sci., Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Acknowledgements:

# http://opencv-python-tutroals.readthedocs.org/en/latest/ \
# py_tutorials/py_calib3d/py_table_of_contents_calib3d/py_table_of_contents_calib3d.html

# http://docs.ros.org/electric/api/cob_camera_calibration/html/calibrator_8py_source.html
# OpenCV 3.0 example - stereo_match.py

# Andy Pound, Durham University, 2016 - calibration save/load approach

#####################################################################

import cv2
import sys
import numpy as np
import os
import argparse
import time
from datetime import datetime
import glob

#####################################################################
# define target framerates in fps (may not be achieved)

calibration_capture_framerate = 2
disparity_processing_framerate = 25
scaleFactor = 0.8

#####################################################################
# wrap different kinds of stereo camera - standard (v4l/vfw), ximea, ZED


class StereoCamera:
    def __init__(self, args):

        self.xiema = args.ximea
        self.zed = args.zed
        self.cameras_opened = False

        if args.ximea:

            # ximea requires specific API offsets in the open commands

            self.camL = cv2.VideoCapture()
            self.camR = cv2.VideoCapture()

            if not(
                (self.camL.open(
                    cv2.CAP_XIAPI)) and (
                    self.camR.open(
                        cv2.CAP_XIAPI +
                    1))):
                print("Cannot open pair of Ximea cameras connected.")
            sys.exit()

        elif args.zed:

            # ZED is a single camera interface with L/R returned as 1 image

            try:
                # to use a non-buffered camera stream (via a separate thread)
                # no T-API use, unless additional code changes later

                import camera_stream
                self.camZED = camera_stream.CameraVideoStream()

            except BaseException:
                # if not then just use OpenCV default

                print("INFO: camera_stream class not found - \
                        camera input may be buffered")
                self.camZED = cv2.VideoCapture()

            if not(self.camZED.open(args.lcamera_to_use)):
                print(
                    "Cannot open connected ZED stereo camera as camera #: ",
                    args.lcamera_to_use)
                sys.exit()

            # report resolution currently in use for ZED (as various
            # options exist) can use .get()/.set() to read/change also

            _, frame = self.camZED.read()
            height, width, channels = frame.shape
            print()
            print("ZED left/right resolution: ",
                  int(width / 2), " x ", int(height))
            print()

        else:

            # by default two standard system connected cams from the default
            # video backend

            try:
                raise BaseException
                import camera_stream
                self.camL = camera_stream.CameraVideoStream(use_tapi=True)
                self.camR = camera_stream.CameraVideoStream(use_tapi=True)
            except BaseException:
                # if not then just use OpenCV default
                print("INFO: camera_stream class not found - \
                        camera input may be buffered")
                self.camL = cv2.VideoCapture()
                self.camR = cv2.VideoCapture()

            cap_backend = cv2.CAP_ANY

            if sys.platform == "win32":
                # use msmf if mjpg because directshow 2048x1536 is not smooth 15fps
                # directshow for all other cases because msmf cannot set camera format (defaults to mjpg)
                if args.camera_format.upper() == "MJPG":
                    cap_backend = cv2.CAP_MSMF
                else:
                    cap_backend = cv2.CAP_DSHOW

            if not (self.camL.open(args.lcamera_to_use, cap_backend) and self.camR.open(args.rcamera_to_use, cap_backend)):
                print(
                    "Cannot open pair of system cameras connected \
                    at camera {} and {}".format(args.lcamera_to_use, args.rcamera_to_use))
                sys.exit()

            self.camL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_format))
            self.camL.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            self.camL.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            self.camL.set(cv2.CAP_PROP_FPS, args.camera_fps)

            self.camR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_format))
            self.camR.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            self.camR.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            self.camR.set(cv2.CAP_PROP_FPS, args.camera_fps)

        self.cameras_opened = True

    def swap_cameras(self):
        if not(self.zed):
            # swap the cameras - for all but ZED camera
            tmp = self.camL
            self.camL = self.camR
            self.camR = tmp

    def get_frames(self):  # return left, right
        if self.zed:

            # grab single frame from camera (read = grab/retrieve)
            # and split into Left and Right

            _, frame = self.camZED.read()
            height, width, channels = frame.shape
            frameL = frame[:, 0:int(width / 2), :]
            frameR = frame[:, int(width / 2):width, :]
        else:
            # grab frames from camera (to ensure best time sync.)

            self.camL.grab()
            self.camR.grab()

            # then retrieve the images in slow(er) time
            # (do not be tempted to use read() !)

            _, frameL = self.camL.retrieve()
            _, frameR = self.camR.retrieve()

        return frameL, frameR


class StereoCameraStub:
    def __init__(self, args):
        self.size = (args.camera_width, args.camera_height, 3)

    def get_frames(self):
        # return np.zeros(self.size, dtype=np.uint8), np.zeros(self.size, dtype=np.uint8)
        return cv2.imread(sorted(glob.glob(args.calibration_images_path + "/*_l.png"))[0]), cv2.imread(sorted(glob.glob(args.calibration_images_path + "/*_r.png"))[0])


def resize_scale(img, scaleFactor):
    return cv2.resize(img, (int(img.shape[1] * scaleFactor), int(img.shape[0] * scaleFactor)))

def remapImageRange(image, newMin = 0.0, newMax = 1.0):
    maxVal = image.max()
    minVal = image.min()
    newImage = (((image - float(minVal)) / float(float(maxVal) - float(minVal))) * float(newMax - newMin)) + float(newMin)
    return newImage


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# good settings
        # self.max_disparity = 256
        # self.blockSize = 101  # or 25

        # self.left_matcher = cv2.StereoBM_create(
        #     numDisparities=self.max_disparity,
        #     blockSize=self.blockSize
        # )
        # self.left_matcher.setTextureThreshold(7)
        # self.left_matcher.setUniquenessRatio(0)
        # self.left_matcher.setDisp12MaxDiff(0)
        # self.left_matcher.setSpeckleRange(3)
        # self.left_matcher.setSpeckleWindowSize(7)
class StereoBM:
    def __init__(self):
        self.max_disparity = 128
        self.blockSize = 25

        self.left_matcher = cv2.StereoBM_create(
            numDisparities=self.max_disparity,
            blockSize=self.blockSize
        )
        self.left_matcher.setTextureThreshold(7)
        self.left_matcher.setUniquenessRatio(0)
        self.left_matcher.setDisp12MaxDiff(0)
        self.left_matcher.setSpeckleRange(3)
        self.left_matcher.setSpeckleWindowSize(7)
        # self.left_matcher.setPreFilterType(1)

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

        # WLS Filter
        self.lambdaVal = 8000.0
        # Might need to adjust sigma if calibration is different and the final filtered image looks completely wrong
        self.sigma = 1.5
        self.ddr = 0.33
        self.depthDisRadius = int(np.ceil(self.ddr * self.blockSize))

        self.wlsFilter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wlsFilter.setLambda(self.lambdaVal)
        self.wlsFilter.setSigmaColor(self.sigma)
        self.wlsFilter.setDepthDiscontinuityRadius(self.depthDisRadius)


    def compute(self, imgL, imgR):
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        left_disparity = self.left_matcher.compute(grayL, grayR)

        right_disparity = self.right_matcher.compute(grayR, grayL)

        # filtered_left_disparity = self.wlsFilter.filter(left_disparity, grayL, disparity_map_right=np.zeros((768,1024,1), dtype=np.uint8))
        filtered_left_disparity = self.wlsFilter.filter(left_disparity, grayL, disparity_map_right=right_disparity)

        return left_disparity, filtered_left_disparity


# good settings
        # self.window_size = 9
        # self.max_disparity = 128

        # self.left_matcher = cv2.StereoSGBM_create(
        #     minDisparity = 0,
        #     numDisparities = self.max_disparity,
        #     blockSize = 7,
        #     P1 = 8*3*self.window_size**2,
        #     P2 = 32*3*self.window_size**2,
        #     # disp12MaxDiff = 0,  # doesn't do anything
        #     # preFilterCap = 0,  # doesn't do anything
        #     # uniquenessRatio = 0, # doesn't do anything
        #     speckleWindowSize = 0,
        #     speckleRange = 0,
        #     mode = cv2.StereoSGBM_MODE_SGBM
        # )
class StereoSGBM:
    def __init__(self):
        self.window_size = 9
        self.max_disparity = 192

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity = 0,
            numDisparities = self.max_disparity,
            blockSize = 7,
            P1 = 8*3*self.window_size**2,
            P2 = 32*3*self.window_size**2,
            # disp12MaxDiff = 100,  # doesn't do anything
            # preFilterCap = 0,  # doesn't do anything
            # uniquenessRatio = 0, # doesn't do anything
            speckleWindowSize = 0,
            speckleRange = 0,
            mode = cv2.StereoSGBM_MODE_SGBM
        )

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

        # WLS Filter
        self.ddr = 0.33
        self.lambdaVal = 8000.0
        # Might need to adjust sigma if calibration is different and the final filtered image looks completely wrong
        self.sigma = 1.5
        self.lrc = 0
        self.depthDisRadius = int(np.ceil(self.ddr * self.window_size))

        self.wlsFilter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wlsFilter.setLambda(self.lambdaVal)
        self.wlsFilter.setSigmaColor(self.sigma)
        # self.wlsFilter.setLRCthresh(self.lrc)
        self.wlsFilter.setDepthDiscontinuityRadius(self.depthDisRadius)

    def compute(self, imgL, imgR):
        # SGBM doesn't require gray images
        # imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        # imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        left_disparity = self.left_matcher.compute(imgL, imgR)
        right_disparity = self.right_matcher.compute(imgR, imgL)

        filtered_left_disparity = self.wlsFilter.filter(left_disparity, imgL, disparity_map_right=right_disparity)

        return left_disparity, filtered_left_disparity



#####################################################################
# deal with optional arguments


def parse_args():
    parser = argparse.ArgumentParser(
        description='Perform full stereo calibration and SGBM matching.')
    parser.add_argument(
        "--ximea",
        help="use a pair of Ximea cameras",
        action="store_true")
    parser.add_argument(
        "--zed",
        help="use a Stereolabs ZED stereo camera",
        action="store_true")
    parser.add_argument(
        "-cl",
        "--lcamera_to_use",
        type=int,
        help="specify left camera to use",
        default=0)
    parser.add_argument(
        "-cr",
        "--rcamera_to_use",
        type=int,
        help="specify right camera to use",
        default=2)
    parser.add_argument(
        "-cw",
        "--camera_width",
        type=int,
        help="camera horizontal resolution",
        default=2048)
    parser.add_argument(
        "-ch",
        "--camera_height",
        type=int,
        help="camera vertical resolution",
        default=1536)
    parser.add_argument(
        "-cf",
        "--camera_format",
        type=str,
        help="camera format",
        default="MJPG")
    parser.add_argument(
        "-dsw",
        "--downsample_width",
        type=int,
        help="downsample (resize) image to this width before processing",
        default=0)
    parser.add_argument(
        "-dsh",
        "--downsample_height",
        type=int,
        help="downsample (resize) image to this height before processing",
        default=0)
    parser.add_argument(
        "-cfps",
        "--camera_fps",
        type=int,
        help="camera fps",
        default=15)
    parser.add_argument(
        "-pr",
        "--preview_ratio",
        type=float,
        help="preview size ratio",
        default=1)
    parser.add_argument(
        "-cbx",
        "--chessboardx",
        type=int,
        help="specify number of internal chessboard squares (corners) \
                in x-direction",
        default=8)
    parser.add_argument(
        "-cby",
        "--chessboardy",
        type=int,
        help="specify number of internal chessboard squares (corners) \
            in y-direction",
        default=6)
    parser.add_argument(
        "-cbw",
        "--chessboardw",
        type=float,
        help="specify width/height of chessboard squares in mm",
        default=25.0)
    parser.add_argument(
        "-co",
        "--calibration_only",
        type=int,
        help="run calibration only, no cameras",
        default=0)
    parser.add_argument(
        "-cp",
        "--calibration_path",
        type=str,
        help="specify path to calibration files to load",
        default="")
    parser.add_argument(
        "-ci",
        "--calibration_images_path",
        type=str,
        help="specify path to load calibration images instead of taking live",
        default="")
    parser.add_argument(
        "-ma",
        "--matching_algorithm",
        type=str,
        help="stereo matching algorithm to use (bm or sgbm)",
        default="bm")
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="specify number of iterations for each stage of optimisation",
        default=30)
    parser.add_argument(
        "-e",
        "--minimum_error",
        type=float,
        help="specify lower error threshold upon which to stop \
                optimisation stages",
        default=0.05)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # handle special cases
    if args.downsample_width or args.downsample_height:
        if not (args.downsample_width and args.downsample_height):
            print("must specify dsw and dsh")
            sys.exit(1)

    #####################################################################

    # flag values to enter processing loops - do not change

    keep_processing = True
    do_calibration = False

    #####################################################################

    # STAGE 1 - open 2 connected cameras

    # define video capture object

    if args.calibration_only:
        stereo_camera = StereoCameraStub(args)
    else:
        stereo_camera = StereoCamera(args)

    # define display window names

    window_nameL = "LEFT Camera Input"  # window name
    window_nameR = "RIGHT Camera Input"  # window name

    # create window by name (as resizable)

    cv2.namedWindow(window_nameL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_nameR, cv2.WINDOW_NORMAL)

    # set sizes and set windows

    frameL, frameR = stereo_camera.get_frames()

    # assuming both cams are same resolution

    # height, width, channels = frameL.shape
    height, width, channels = args.camera_height, args.camera_width, 3
    if args.downsample_width:
        height = args.downsample_height
        width = args.downsample_width
    IMAGE_SIZE = (width, height)
    PREVIEW_SIZE = (int(height * args.preview_ratio), int(width * args.preview_ratio))

    height_prev, width_prev = PREVIEW_SIZE
    cv2.resizeWindow(window_nameL, width_prev, height_prev)

    # height, width, channels = frameR.shape
    # height_prev, width_prev = int(height * args.preview_ratio), int(width * args.preview_ratio)
    cv2.resizeWindow(window_nameR, width_prev, height_prev)

    # controls

    print("s : swap cameras left and right")
    print("e : export camera calibration to file")
    print("l : load camera calibration from file")
    print("x : exit")
    print()
    print("space : continue to next stage")
    print()

    while (keep_processing):

        # get frames from camera

        frameL, frameR = stereo_camera.get_frames()
        if args.downsample_width:
            frameL = cv2.resize(frameL, (args.downsample_width, args.downsample_height), frameL, cv2.INTER_LINEAR)
            frameR = cv2.resize(frameR, (args.downsample_width, args.downsample_height), frameR, cv2.INTER_LINEAR)

        # frameL_prev = cv2.resize(frameL, (width_prev, height_prev))
        # frameR_prev = cv2.resize(frameR, (width_prev, height_prev))

        # display image

        cv2.imshow(window_nameL, frameL)
        cv2.imshow(window_nameR, frameR)
        # cv2.imshow(window_nameL, frameL_prev)
        # cv2.imshow(window_nameR, frameR_prev)

        # start the event loop - essential

        key = cv2.waitKey(40) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # loop control - space to continue; x to exit; s - swap cams; l - load

        if (key == ord(' ')):
            keep_processing = False
        elif (key == ord('x')):
            sys.exit()
        elif (key == ord('s')):
            # swap the cameras if specified

            stereo_camera.swap_cameras()
        elif (key == ord('l')):

            if (not args.calibration_path):
                print("Error - no calibration path specified:")
                sys.exit()

            # load calibration from file

            os.chdir(args.calibration_path)
            print('Using calibration files: ', args.calibration_path)
            mapL1 = np.load('mapL1.npy')
            mapL2 = np.load('mapL2.npy')
            mapR1 = np.load('mapR1.npy')
            mapR2 = np.load('mapR2.npy')
            print("mapL1 first", mapL1[0][0])
            print("mapL2 first", mapL2[0][0])
            print("mapR1 first", mapR1[0][0])
            print("mapR2 first", mapR2[0][0])
            os.chdir("../")

            keep_processing = False
            do_calibration = True  # set to True to skip next loop

    #####################################################################

    # STAGE 2: perform intrinsic calibration (removal of image distortion in
    # each image)

    termination_criteria_subpix = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        args.iterations,
        args.minimum_error)

    # set up a set of real-world "object points" for the chessboard pattern

    patternX = args.chessboardx
    patternY = args.chessboardy
    square_size_in_mm = args.chessboardw

    if (patternX == patternY):
        print("*****************************************************************")
        print()
        print("Please use a chessboard pattern that is not equal dimension")
        print("in X and Y (otherwise a rotational ambiguity exists!).")
        print()
        print("*****************************************************************")
        print()
        sys.exit()

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

    objp = np.zeros((patternX * patternY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternX, 0:patternY].T.reshape(-1, 2)
    objp = objp * square_size_in_mm

    # create arrays to store object points and image points from all the images

    # ... both for paired chessboard detections (L AND R detected)

    objpoints_pairs = []         # 3d point in real world space
    imgpoints_right_paired = []  # 2d points in image plane.
    imgpoints_left_paired = []   # 2d points in image plane.

    # ... and for left and right independantly (L OR R detected, OR = logical OR)

    objpoints_left_only = []   # 3d point in real world space
    imgpoints_left_only = []   # 2d points in image plane.

    objpoints_right_only = []   # 3d point in real world space
    imgpoints_right_only = []   # 2d points in image plane.

    # count number of chessboard detection (across both images)

    chessboard_pattern_detections_paired = 0
    chessboard_pattern_detections_left = 0
    chessboard_pattern_detections_right = 0

    print()
    print("--> hold up chessboard")
    print("press space when ready to start calibration stage  ...")
    print()

    if args.calibration_images_path:
        print("using saved images for calibration")
        image_paths_left = sorted(glob.glob(args.calibration_images_path + "/*_l.png"))
        image_paths_right = sorted(glob.glob(args.calibration_images_path + "/*_r.png"))

    while (not(do_calibration)):

        # get frames from camera

        if args.calibration_images_path:
            if image_paths_left and image_paths_right:
                frameL = cv2.imread(image_paths_left.pop())
                frameR = cv2.imread(image_paths_right.pop())
            else:
                do_calibration = True
                continue
        else:
            frameL, frameR = stereo_camera.get_frames()

        if args.downsample_width:
            frameL = cv2.resize(frameL, (args.downsample_width, args.downsample_height), frameL, cv2.INTER_LINEAR)
            frameR = cv2.resize(frameR, (args.downsample_width, args.downsample_height), frameR, cv2.INTER_LINEAR)

        # blurL, blurR = cv2.Laplacian(frameL, cv2.CV_64F).var(), cv2.Laplacian(frameR, cv2.CV_64F).var()

        # print("blur L: {}    blur R: {}".format(blurL, blurR))

        # convert to grayscale

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners in the images
        # (change flags to perhaps improve detection - see OpenCV manual)

        retR, cornersR = cv2.findChessboardCorners(
            grayR, (patternX, patternY), None, cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        retL, cornersL = cv2.findChessboardCorners(
            grayL, (patternX, patternY), None, cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

        # when found, add object points, image points (after refining them)

        # N.B. to allow for maximal coverage of the FoV of the L and R images
        # for instrinsic calc. without an image region being underconstrained
        # record and process detections for 3 conditions in 3 differing list
        # structures

        # -- > detected in left (only or also in right)

        if (retL):

            chessboard_pattern_detections_left += 1

            # add object points to left list

            objpoints_left_only.append(objp)

            # refine corner locations to sub-pixel accuracy and then add to list

            corners_sp_L = cv2.cornerSubPix(
                grayL, cornersL, (11, 11), (-1, -1), termination_criteria_subpix)
            imgpoints_left_only.append(corners_sp_L)

        # -- > detected in right (only or also in left)

        if (retR):

            chessboard_pattern_detections_right += 1

            # add object points to left list

            objpoints_right_only.append(objp)

            # refine corner locations to sub-pixel accuracy and then add to list

            corners_sp_R = cv2.cornerSubPix(
                grayR, cornersR, (11, 11), (-1, -1), termination_criteria_subpix)
            imgpoints_right_only.append(corners_sp_R)

        # -- > detected in left and right

        if ((retR) and (retL)):

            chessboard_pattern_detections_paired += 1

            # add object points to global list

            objpoints_pairs.append(objp)

            # add previously refined corner locations to list

            imgpoints_left_paired.append(corners_sp_L)
            imgpoints_right_paired.append(corners_sp_R)

        # display detections / chessboards

        text = 'detected L: ' + str(chessboard_pattern_detections_left) + \
            ' detected R: ' + str(chessboard_pattern_detections_right)
        cv2.putText(frameL, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)
        text = 'detected (L AND R): ' + str(chessboard_pattern_detections_paired)
        cv2.putText(frameL, text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)

        # draw the corners / chessboards

        drawboardL = cv2.drawChessboardCorners(
                                frameL, (patternX, patternY), cornersL, retL)
        drawboardR = cv2.drawChessboardCorners(
                                frameR, (patternX, patternY), cornersR, retR)
        cv2.imshow(window_nameL, drawboardL)
        cv2.imshow(window_nameR, drawboardR)

        # start the event loop

        key = cv2.waitKey(int(1000 / calibration_capture_framerate)) & 0xFF
        if (key == ord(' ')):
            do_calibration = True
        elif (key == ord('x')):
            sys.exit()

    # perform calibration on both cameras - uses [Zhang, 2000]

    termination_criteria_intrinsic = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        args.iterations,
        args.minimum_error)

    if (chessboard_pattern_detections_paired > 0):  # i.e. if did not load calib.

        print("START - intrinsic calibration ...")

        # rms_int_L, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
        #     objpoints_left_only, imgpoints_left_only, IMAGE_SIZE,
        #     None, None, criteria=termination_criteria_intrinsic)
        # rms_int_R, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
        #     objpoints_right_only, imgpoints_right_only, IMAGE_SIZE,
        #     None, None, criteria=termination_criteria_intrinsic)
        rms_int_L, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
            objpoints_pairs, imgpoints_left_paired, IMAGE_SIZE,
            None, None, criteria=termination_criteria_intrinsic)
        rms_int_R, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
            objpoints_pairs, imgpoints_right_paired, IMAGE_SIZE,
            None, None, criteria=termination_criteria_intrinsic)
        print("FINISHED - intrinsic calibration")

        print()
        print("LEFT: RMS left intrinsic calibration re-projection error: ",
            rms_int_L)
        print("RIGHT: RMS right intrinsic calibration re-projection error: ",
            rms_int_R)
        print()

        # perform undistortion of the images

        keep_processing = True

        print()
        print("-> displaying undistortion")
        print("press space to continue to next stage ...")
        print()

    while (keep_processing):

        # get frames from camera

        frameL, frameR = stereo_camera.get_frames()
        if args.downsample_width:
            frameL = cv2.resize(frameL, (args.downsample_width, args.downsample_height), frameL, cv2.INTER_LINEAR)
            frameR = cv2.resize(frameR, (args.downsample_width, args.downsample_height), frameR, cv2.INTER_LINEAR)

        undistortedL = cv2.undistort(frameL, mtxL, distL, None, None)
        undistortedR = cv2.undistort(frameR, mtxR, distR, None, None)

        # display image

        cv2.imshow(window_nameL, undistortedL)
        cv2.imshow(window_nameR, undistortedR)

        # start the event loop - essential

        key = cv2.waitKey(int(1000 / disparity_processing_framerate)) & 0xFF

        # loop control - space to continue; x to exit

        if (key == ord(' ')):
            keep_processing = False
        elif (key == ord('x')):
            sys.exit()

    # show mean re-projection error of the object points onto the image(s)

    if (chessboard_pattern_detections_paired > 0):  # i.e. if did not load a calib.

        tot_errorL = 0
        for i in range(len(objpoints_pairs)):
            imgpoints_left_paired2, _ = cv2.projectPoints(
                objpoints_pairs[i], rvecsL[i], tvecsL[i], mtxL, distL)
            errorL = cv2.norm(
                imgpoints_left_paired[i],
                imgpoints_left_paired2,
                cv2.NORM_L2) / len(imgpoints_left_paired2)
            tot_errorL += errorL

        print("LEFT: mean re-projection error (absolute, px): ",
            tot_errorL / len(objpoints_pairs))

        tot_errorR = 0
        for i in range(len(objpoints_pairs)):
            imgpoints_right_paired2, _ = cv2.projectPoints(
                objpoints_pairs[i], rvecsR[i], tvecsR[i], mtxR, distR)
            errorR = cv2.norm(
                imgpoints_right_paired[i],
                imgpoints_right_paired2,
                cv2.NORM_L2) / len(imgpoints_right_paired2)
            tot_errorR += errorR

        print("RIGHT: mean re-projection error (absolute, px): ",
            tot_errorR / len(objpoints_pairs))

    #####################################################################

    # STAGE 3: perform extrinsic calibration (recovery of relative camera
    # positions)

    # this takes the existing calibration parameters used to undistort the
    # individual images as well as calculated the relative camera positions
    # - represented via the fundamental matrix, F

    # alter termination criteria to (perhaps) improve solution - ?

    termination_criteria_extrinsics = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        args.iterations,
        args.minimum_error)

    if (chessboard_pattern_detections_paired > 0):  # i.e. if did not load a calib.
        print()
        print("START - extrinsic calibration ...")
        (rms_stereo,
        camera_matrix_l,
        dist_coeffs_l,
        camera_matrix_r,
        dist_coeffs_r,
        R,
        T,
        E,
        F) = cv2.stereoCalibrate(objpoints_pairs,
                                imgpoints_left_paired,
                                imgpoints_right_paired,
                                mtxL,
                                distL,
                                mtxR,
                                distR,
                                grayL.shape[::-1],
                                criteria=termination_criteria_extrinsics,
                                flags=0)

        print("FINISHED - extrinsic calibration")

        print()
        print("Intrinsic Camera Calibration:")
        print()
        print("Intrinsic Camera Calibration Matrix, K - from \
                intrinsic calibration:")
        print("(format as follows: fx, fy - focal lengths / cx, \
                cy - optical centers)")
        print("[fx, 0, cx]\n[0, fy, cy]\n[0,  0,  1]")
        print()
        print("Intrinsic Distortion Co-effients, D - from intrinsic calibration:")
        print("(k1, k2, k3 - radial p1, p2 - tangential distortion coefficients)")
        print("[k1, k2, p1, p2, k3]")
        print()
        print("K (left camera)")
        print(camera_matrix_l)
        print("distortion coeffs (left camera)")
        print(dist_coeffs_l)
        print()
        print("K (right camera)")
        print(camera_matrix_r)
        print("distortion coeffs (right camera)")
        print(dist_coeffs_r)

        print()
        print("Extrinsic Camera Calibration:")
        print("Rotation Matrix, R (left -> right camera)")
        print(R)
        print()
        print("Translation Vector, T (left -> right camera)")
        print(T)
        print()
        print("Essential Matrix, E (left -> right camera)")
        print(E)
        print()
        print("Fundamental Matrix, F (left -> right camera)")
        print(F)

        print()
        print("STEREO: RMS left to  right re-projection error: ", rms_stereo)

    #####################################################################

    # STAGE 4: rectification of images (make scan lines align left <-> right

    # N.B.  "alpha=0 means that the rectified images are zoomed and shifted so that
    # only valid pixels are visible (no black areas after rectification).
    # alpha=1 means that the rectified image is decimated and shifted so that
    # all the pixels from the original images from the cameras are retained
    # in the rectified images (no source image pixels are lost)." - ?
    print("chessboard_pattern_detections_paired", chessboard_pattern_detections_paired)
    if (chessboard_pattern_detections_paired > 0):  # i.e. if did not load calib.
        RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
            camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r,
            grayL.shape[::-1], R, T, alpha=-1)

    # compute the pixel mappings to the rectified versions of the images

    if (chessboard_pattern_detections_paired > 0):  # i.e. if did not load calib.
        mapL1, mapL2 = cv2.initUndistortRectifyMap(
            camera_matrix_l, dist_coeffs_l, RL, PL, grayL.shape[::-1],
            cv2.CV_32FC1)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(
            camera_matrix_r, dist_coeffs_r, RR, PR, grayR.shape[::-1],
            cv2.CV_32FC1)

        print()
        print("-> displaying rectification")
        print("press space to continue to next stage ...")

        keep_processing = True

    while (keep_processing):

        # get frames from camera

        frameL, frameR = stereo_camera.get_frames()
        if args.downsample_width:
            frameL = cv2.resize(frameL, (args.downsample_width, args.downsample_height), frameL, cv2.INTER_LINEAR)
            frameR = cv2.resize(frameR, (args.downsample_width, args.downsample_height), frameR, cv2.INTER_LINEAR)

        # undistort and rectify based on the mappings (could improve interpolation
        # and image border settings here)

        undistorted_rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
        undistorted_rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

        # display image

        cv2.imshow(window_nameL, undistorted_rectifiedL)
        cv2.imshow(window_nameR, undistorted_rectifiedR)

        # start the event loop - essential

        key = cv2.waitKey(int(1000 / disparity_processing_framerate)) & 0xFF

        # loop control - space to continue; x to exit

        if (key == ord(' ')):
            keep_processing = False
        elif (key == ord('x')):
            sys.exit()

    #####################################################################

    # STAGE 5: calculate stereo depth information

    # uses a modified H. Hirschmuller algorithm [HH08] that differs (see
    # opencv manual)

    # parameters can be adjusted, current ones from [Hamilton / Breckon et al.
    # 2013] - numDisparities=128, SADWindowSize=21)

    print()
    print("-> display disparity image")
    print("press x to exit")
    print("press e to export calibration")
    print("press c for false colour mapped disparity")
    print("press f for fullscreen disparity")

    print()

    # set up defaults for disparity calculation

    if args.matching_algorithm == "sgbm":
        bm = StereoSGBM()
    else:
        # BM by default
        bm = StereoBM()
    print("using stereo matching algorithm:", args.matching_algorithm)

    keep_processing = True

    # set up disparity window to be correct size


    window_nameD = "Stereo Disparity"
    cv2.namedWindow(window_nameD, cv2.WINDOW_NORMAL)
    height, width, channels = args.camera_height, args.camera_width, 3
    cv2.resizeWindow(window_nameD, *(PREVIEW_SIZE[::-1]))


    window_nameDF = "Stereo Disparity Filtered"
    cv2.namedWindow(window_nameDF, cv2.WINDOW_NORMAL)
    height, width, channels = args.camera_height, args.camera_width, 3
    cv2.resizeWindow(window_nameDF, *(PREVIEW_SIZE[::-1]))

    while (keep_processing):

        # get frames from camera

        origFrameL, origFrameR = stereo_camera.get_frames()
        frameL, frameR = origFrameL, origFrameR
        if args.downsample_width:
            frameL = cv2.resize(frameL, (args.downsample_width, args.downsample_height), frameL, cv2.INTER_LINEAR)
            frameR = cv2.resize(frameR, (args.downsample_width, args.downsample_height), frameR, cv2.INTER_LINEAR)

        # frameL = cv2.fastNlMeansDenoisingColored(frameL,None,6,6,7,21)
        # frameR = cv2.fastNlMeansDenoisingColored(frameR,None,6,6,7,21)

        # undistort and rectify based on the mappings (could improve interpolation
        # and image border settings here)
        # N.B. mapping works independant of number of image channels

        undistorted_rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
        undistorted_rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)
        
        undistorted_rectifiedL = adjust_gamma(undistorted_rectifiedL, 1.4)
        undistorted_rectifiedR = adjust_gamma(undistorted_rectifiedR, 1.4)
        


        # compute disparity image from undistorted and rectified versions
        # (which for reasons best known to the OpenCV developers is returned
        # scaled by 16)

        disparity, disparity_filtered = bm.compute(undistorted_rectifiedL, undistorted_rectifiedR)

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 -> max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(
            disparity, 0, bm.max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        _, disparity_filtered = cv2.threshold(
            disparity_filtered, 0, bm.max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_filtered_scaled = (disparity_filtered / 16.).astype(np.uint8)

        # display image

        grayL = cv2.cvtColor(undistorted_rectifiedL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(undistorted_rectifiedR, cv2.COLOR_BGR2GRAY)
        cv2.imshow(window_nameL, grayL)
        cv2.imshow(window_nameR, grayR)

        # depth_map = cv2.reprojectImageTo3D(disparity_filtered_scaled, Q)
        # depth_map_z = depth_map[:,:,2]

        # max_val = depth_map_z[np.isfinite(depth_map_z)].max()
        # print("max val = ", max_val)
        # np.nan_to_num(depth_map_z, False, max_val, max_val, max_val)
        # print(depth_map_z)

        # depth_map_z2 = np.zeros((768, 1024), dtype=np.uint8)
        # depth_map_z2 = cv2.normalize(depth_map_z, None, 0, 255, cv2.NORM_MINMAX)
        # depth_map_z2 = depth_map_z2.astype(np.uint8)
        # print(depth_map_z2)

        focallength = 2.15
        baseline = 144
        scaling_constant = 0.7

        np.nan_to_num(disparity_filtered, False, 0, 0, 0)
        print("disparity map:", cv2.resize(disparity_filtered, (12, 16)))
        disparity_filtered_disp = cv2.normalize(disparity_filtered, None, 0, 255, cv2.NORM_MINMAX)
        disparity_filtered_disp = disparity_filtered_disp.astype(np.uint8)
        print("disparity map 2:", cv2.resize(disparity_filtered_disp, (12, 16)))
        cv2.imshow("raw disparity", disparity_filtered_disp)

        depth_map = (focallength * baseline * scaling_constant * 16) / disparity_filtered
        np.nan_to_num(depth_map, False, 0, 0, 0)
        print("depth map:", cv2.resize(depth_map, (12, 16)))

        # depth_map_disp = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        _, depth_map_disp = cv2.threshold(depth_map, 12.25, 0, cv2.THRESH_TOZERO_INV, None)
        depth_map_disp = depth_map_disp * 20
        depth_map_disp = depth_map_disp.astype(np.uint8)
        # print("depth map disp:", cv2.resize(depth_map_disp, (12, 16)))

        print("depth map 2:", cv2.resize(depth_map_disp, (12, 16)))
        cv2.imshow("depth map distance", depth_map_disp)

        # display disparity - which ** for display purposes only ** we re-scale to
        # 0 ->255

        cv2.imshow(window_nameD, (disparity_scaled *
                                (256. / bm.max_disparity)).astype(np.uint8))

        cv2.imshow(window_nameDF, (disparity_filtered_scaled *
                                (256. / bm.max_disparity)).astype(np.uint8))

        # start the event loop - essential

        key = cv2.waitKey(int(1000 / disparity_processing_framerate)) & 0xFF

        # loop control - x to exit

        if (key == ord(' ')):
            keep_processing = False
        elif (key == ord('x')):
            sys.exit()
        elif (key == ord('f')):
            cv2.setWindowProperty(
                window_nameD,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)
        elif (key == ord('s')):
            # save all images
            folderName = datetime.now().strftime('capture_%d-%m-%y-%H-%M-%S-%f')
            print(os.getcwd())
            os.mkdir(folderName)
            os.chdir(folderName)
            with open("settings.txt", "w") as f:
                print("calibration folder: " + args.calibration_path, file=f)
            cv2.imwrite("original_left.png", origFrameL)
            cv2.imwrite("original_right.png", origFrameR)
            cv2.imwrite("undistorted_left.png", undistorted_rectifiedL)
            cv2.imwrite("undistorted_right.png", undistorted_rectifiedR)
            cv2.imwrite("disparity_raw_left.png", disparity_scaled)
            cv2.imwrite("disparity_filtered_left.png", disparity_filtered_scaled)
            cv2.imwrite("disparity_filtered_display_left.png", disparity_filtered_disp)
            cv2.imwrite("depth_left.png", depth_map)
            cv2.imwrite("depth_display_left.png", depth_map_disp)
            print("saved stereo images")
        elif (key == ord('e')):
            # export to named folder path as numpy data
            try:
                os.mkdir('calibration')
            except OSError:
                print("Exporting to existing calibration archive directory.")
            os.chdir('calibration')
            folderName = time.strftime('%d-%m-%y-%H-%M-rms-') + \
                ('%.2f' % rms_stereo) + '-zed-' + str(int(args.zed)) \
                + '-ximea-' + str(int(args.ximea))
            os.mkdir(folderName)
            os.chdir(folderName)
            np.save('mapL1', mapL1)
            np.save('mapL2', mapL2)
            np.save('mapR1', mapR1)
            np.save('mapR2', mapR2)
            cv_file = cv2.FileStorage("calibration.xml", cv2.FILE_STORAGE_WRITE)
            cv_file.write("source", ' '.join(sys.argv[0:]))
            cv_file.write(
                "description",
                "camera matrices K for left and right, distortion coefficients " +
                "for left and right, 3D rotation matrix R, 3D translation " +
                "vector T, Essential matrix E, Fundamental matrix F, disparity " +
                "to depth projection matrix Q")
            cv_file.write("K_l", camera_matrix_l)
            cv_file.write("K_r", camera_matrix_r)
            cv_file.write("distort_l", dist_coeffs_l)
            cv_file.write("distort_r", dist_coeffs_r)
            cv_file.write("R", R)
            cv_file.write("T", T)
            cv_file.write("E", E)
            cv_file.write("F", F)
            cv_file.write("Q", Q)
            cv_file.release()
            print("Exported to path: ", folderName)
            print("saving F32 BE and LE files")
            mapL1.astype(">f4").tofile("mapL1_BE.wav")
            mapL2.astype(">f4").tofile("mapL2_BE.wav")
            mapR1.astype(">f4").tofile("mapR1_BE.wav")
            mapR2.astype(">f4").tofile("mapR2_BE.wav")
            mapL1.astype("f4").tofile("mapL1_LE.wav")
            mapL2.astype("f4").tofile("mapL2_LE.wav")
            mapR1.astype("f4").tofile("mapR1_LE.wav")
            mapR2.astype("f4").tofile("mapR2_LE.wav")

            Q.astype(">f4").tofile("Q_BE.wav")
            Q.astype("f4").tofile("Q_LE.wav")
            print("done")

    #####################################################################

    # close all windows and cams.

    cv2.destroyAllWindows()

    #####################################################################

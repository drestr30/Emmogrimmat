from typing import Union

import cv2 as cv
from os.path import realpath, normpath
import numpy as np
import Filtro

# img = cv.imread('/Users/david/Desktop/Grimmanet/Images/Other1.jpg')


class Detection:

    def __init__(self):
        self.eyes_cascades = None
        self.mouth_cascades = None
        self.prototxt = None
        self.model = None
        self.net = None

    def load_CNN_detector(self):
        self.prototxt = "/home/fourier/Documentos/David Restrepo/Avanzado 2/DL_face_detection/deploy.prototxt.txt"
        self.model = "/home/fourier/Documentos/David Restrepo/Avanzado 2/DL_face_detection/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv.dnn.readNetFromCaffe(self.prototxt, self.model)

    def CNN_face_detection(self, image):
        (ih, iw) = image.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions
        # print("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward()

        confidence = detections[0, 0, 1, 2]

        # print("confidence: {:.2f}%".format(confidence.any() * 100))

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.99:
            raise ValueError("No valid faces detected")
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, 1, 3:7] * np.array([iw, ih, iw, ih])
        (startX, startY, endX, endY) = box.astype("int")
        face = (startX, startY, endX - startX, endY - startY)
        return face

    def load_cascades(self):

        #Eyes initializze
        eyes_cascade_xml = ['haarcascade_eye.xml', ('haarcascade_righteye_2splits.xml', 'haarcascade_lefteye_2splits.xml')]
                                         # 'haarcascade_eye_tree_eyeglasses.xml', # 'haarcascade_mcs_eyepair_big.xml',]
        eye_cascade_paths = (normpath(realpath(cv.__file__) + '/../data/' + eyes_cascade_xml[0]),
                         normpath(realpath(cv.__file__) + '/../data/' + eyes_cascade_xml[1][0]),
                         normpath(realpath(cv.__file__) + '/../data/' + eyes_cascade_xml[1][1]))
        self.eyes_cascades = [cv.CascadeClassifier(eye_cascade_paths[0]),
                    (cv.CascadeClassifier(eye_cascade_paths[1]), cv.CascadeClassifier(eye_cascade_paths[2]))]

        #Mouth initialize
        mouth_cascade_xml = ['haarcascade_mcs_mouth.xml','haarcascade_smile.xml']
        mouth_cascade_paths = (normpath(realpath(cv.__file__) + '/../data/' + mouth_cascade_xml[0]),
                               normpath(realpath(cv.__file__) + '/../data/' + mouth_cascade_xml[1]))
        self.mouth_cascades = [cv.CascadeClassifier(mouth_cascade_paths[0]), cv.CascadeClassifier(mouth_cascade_paths[1])]



    def eyes_detection(self, img):

        for eyes_cascade in self.eyes_cascades:

            if isinstance(eyes_cascade, tuple):

                eyes_r = eyes_cascade[0].detectMultiScale(img, 1.3, 5)
                eyes_l = eyes_cascade[1].detectMultiScale(img, 1.3, 5)

                if type(eyes_r) == np.ndarray and type(eyes_l) == np.ndarray:
                    eyes = np.concatenate((eyes_r, eyes_l))
                elif type(eyes_r) == tuple or type(eyes_l) == tuple:
                    if type(eyes_r) == tuple:
                        eyes = eyes_l
                    if type(eyes_l) == tuple:
                        eyes = eyes_r

            else:

                eyes = eyes_cascade.detectMultiScale(img, 1.3, 5)
            # minSize=(int(round(w/5.5)), int(round(w/10.5))),
            # maxSize=(int(round(w/4.5)), int(round(w/9.5))))


            if isinstance(eyes, tuple) and not eyes:
                # print("No eyes recognized")
                continue
            else:
                # print("Eyes recognized with cascade:", self.eyes_cascades.index(eyes_cascade))
                break

        return eyes

    def mouth_detection(self, img, face=()):

        for mouth_cascade in self.mouth_cascades:

            if face:
                (x, y, w, h) = face
                mouth = mouth_cascade.detectMultiScale(img, 1.3, 5,
                                               minSize=(int(round(w / 3.5)), int(round(h / 6.5))),
                                               maxSize=(int(round(3 * w / 2.5)), int(round(h / 3))))
            else:
                mouth = mouth_cascade.detectMultiScale(img, 1.3, 5)

            if isinstance(mouth, tuple) and not mouth:
                # print("No mouth recognized")
                continue
            else:
                # print("Mouth recognized with cascade:", self.mouth_cascades.index(mouth_cascade))
                break
    #
    # for (x, y, w, h) in eyes:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return mouth

    def create_segment_mask(self, _face, _mouth, _mouthBB, _eyes, _eyesBB):

        # if not isinstance(smile, tuple) and not isinstance(eyes, tuple):  ## if eyes and mouth detected
            ## segment face
        w = _face[2]

        mouth_max_index = np.where(_mouth[:, 2] == _mouth[:, 2].max())[0][0]  ## Choose the widthest mouth
        (x1r, y1r) = [_mouthBB[0] + _mouth[mouth_max_index, 0] + _mouth[mouth_max_index, 2],
                      _mouthBB[1] + _mouth[mouth_max_index, 1] + _mouth[mouth_max_index, 3]]
        (x1l, y1l) = [_mouthBB[0] + _mouth[mouth_max_index, 0],
                      _mouthBB[1] + _mouth[mouth_max_index, 1] + _mouth[mouth_max_index, 3]]
        try:
            eye_right_index = np.where(_eyes[:, 0] == _eyes[:, 0].max())[0][0]
            (_x2r, _y2r) = [_eyesBB[0] + _eyes[eye_right_index, 0] + _eyes[eye_right_index, 2],
                            _eyesBB[1] + _eyes[eye_right_index, 1] + _eyes[eye_right_index, 3]]
            y3r = _eyesBB[1] + _eyes[eye_right_index, 1] - _eyes[eye_right_index, 3]
            x3r = (_eyes[0, 1] - y1r) * (_x2r - x1r) / (_y2r - y1r) + x1r
        except IndexError:
            (x3r, y3r) = (0, 0)

        try:
            eye_left_index = np.where(_eyes[:, 0] == _eyes[:, 0].min())[0][0]
            (_x2l, _y2l) = [_eyes[eye_left_index, 0],
                            _eyesBB[1] + _eyes[eye_left_index, 1] + _eyes[eye_left_index, 3]]
            y3l = _eyesBB[1] + _eyes[eye_left_index, 1] - _eyes[eye_left_index, 3]
            x3l = (_eyes[1, 1] - y1l) * (_x2l - x1l) / (_y2l - y1l) + x1l
        except IndexError:
            (x3l, y3l) = (0, 0)

        if x3r == 0 or x3r < w / 2:
            (x3r, y3r) = (w - x3l, y3l)

        if x3l == 0 or x3l > w / 2:
            (x3l, y3l) = (w - x3r, y3r)

        vertices = np.array([[x1r, y1r], [x1l, y1l], [x3r, y3r], [x3l, y3l]], dtype=np.int32)
        vertices = cv.convexHull(vertices)

        return vertices


    # def filter_homo(self, img):
    #
    #     img_YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    #
    #     Y_img, Cr, Cb = cv.split(img_YCrCb)
    #
    #     img_log = np.log1p(np.array(Y_img))
    #     img_fft = Filtro.fourier_transform(img_log)
    #
    #     rows, cols, dim = img_YCrCb.shape
    #     rh, rl, cutoff = 1.5, 0.3, 32
    #
    #     DX = cutoff
    #     G = np.ones((rows, cols))
    #     for i in range(rows):
    #         for j in range(cols):
    #             G[i][j] = ((rh - rl) * (1 - np.exp(-((i - rows / 2) ** 2 + (j - cols / 2) ** 2) / (2 * DX ** 2)))) + rl
    #
    #     result_filter = np.multiply(G, img_fft)
    #     result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))
    #
    #     result_exp = np.exp(result_interm)
    #     result = np.array((result_exp / result_exp.max()) * 255, dtype=np.uint8)
    #     result = cv.equalizeHist(result)
    #     return result

# face_detection(img)
# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()


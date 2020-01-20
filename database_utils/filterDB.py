from os.path import realpath, normpath
import cv2 as cv
from detection import Detection
import numpy as np
import time
import os
import Filtro

start = time.time()

### input image
path = '/home/fourier/Documentos/Datasets_emociones/FERG/FERG_ordered'
out_path = '/home/fourier/Documentos/Datasets_emociones/FERG/FERG_faces'

#initialize detector
detector = Detection()
detector.load_cascades()
detector.load_CNN_detector()

for emotion in os.listdir(path):
    if not emotion == "Files.txt" :
        if emotion == "fear":
            for image in os.listdir(os.path.join(path,emotion)):
                if not image.startswith('.') and image.__contains__('ray'):

                    img = cv.imread(os.path.join(path,emotion,image))
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                    # gray = cv.cvtColor(fil_img, cv.COLOR_BGR2GRAY)

                    try:
                        face = detector.CNN_face_detection(img)
                    except ValueError:
                        continue

                    (x,y,w,h) = face

                    cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)

                    # cv.imshow('img', gray)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()

                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]

                    #Eyes detection
                    #
                    # eyes_roiBB = [0, int(h/6), w, int(h/2.5)]
                    # # cv.rectangle(roi_color, (eyes_roiBB[0], eyes_roiBB[1]), (eyes_roiBB[0]+eyes_roiBB[2], eyes_roiBB[1]+eyes_roiBB[3]), (0, 255, 120), 2)
                    # roi_eyes = roi_gray[eyes_roiBB[1]:eyes_roiBB[1]+eyes_roiBB[3], eyes_roiBB[0]:eyes_roiBB[0]+eyes_roiBB[2]]
                    # eyes = detector.eyes_detection(roi_eyes)
                    # # for (ex,ey,ew,eh) in eyes:
                    #     # cv.rectangle(roi_eyes,(ex,ey),(ex+ew,ey+eh),(123,255,123),2)
                    #     # seg_img[eyes_roiBB[1]+ey:(eyes_roiBB[1]+ey+eh), (eyes_roiBB[0]+ex):(eyes_roiBB[0]+ex+ew)] = roi_eyes[ey:ey+eh, ex:ex+ew]
                    #
                    # #Mouth detection
                    #
                    # smile_roiBB = [int(w / 6), int(h / 2), int(4* w / 6), int(h / 2)]
                    # # cv.rectangle(roi_color, (smile_roiBB[0], smile_roiBB[1]), (smile_roiBB[0]+smile_roiBB[2], smile_roiBB[1]+smile_roiBB[3]), (0, 255, 120), 2)
                    # roi_smile = roi_gray[smile_roiBB[1]:smile_roiBB[1]+smile_roiBB[3], smile_roiBB[0]:smile_roiBB[0]+smile_roiBB[2]]
                    # smile = detector.mouth_detection(roi_smile, (x,y,w,h))
                    # # for (sx,sy,sw,sh) in smile:
                    #     # cv.rectangle(roi_smile, (sx,sy), (sx+sw,sy+sh), (0, 0, 255), 2)
                    #     # seg_img[(smile_roiBB[1]+sy):(smile_roiBB[1]+sy)+sh, smile_roiBB[0]+sx:smile_roiBB[0]+sx+sw] = roi_smile[sy:sy + sh, sx:sx + sw]
                    #
                    #
                    #
                    # mask = np.zeros_like(roi_gray)
                    # try:
                    #     global vertices
                    #     vertices = detector.create_segment_mask(face, smile, smile_roiBB, eyes, eyes_roiBB)
                    #     print("Segment mask actualized")
                    #
                    # except TypeError:
                    #     print("Segment mask not actualized, not eyes and mouth detected")
                    #     continue #pass for video
                    #
                    # try:
                    #     cv.fillConvexPoly(mask, vertices, 255)
                    # except NameError:
                    #     print("No vertices recognized, eyes and mouth not detected ")

                        #cv.imshow('img', masked)
                        #cv.waitKey(0)
                        #cv.destroyAllWindows()

                        # LG filter stage
                        # width, height = roi_gray.shape
                        #
                        # kernel = Filtro.laguerre_gauss_filter(height, width, 0.8)
                        # lenna_fourier = Filtro.fourier_transform(roi_gray)
                        # kernel_fourier = Filtro.fourier_transform(kernel)
                        #
                        # out = np.multiply(kernel_fourier, lenna_fourier)
                        # out = np.abs(Filtro.inverse_fourier_transform(out))
                        #
                        # fil_img = out / out.max()  # normalize the data to 0 - 1
                        # fil_img = 255 * fil_img  # Now scal by 255
                        # fil_img = fil_img.astype(np.uint8)

                    # masked = cv.bitwise_and(roi_gray, mask)

                    # cv.imshow("filtered", masked)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()

                    out_path_emotion = os.path.join(out_path,emotion)
                    if not os.path.exists(out_path_emotion):
                        os.mkdir(out_path_emotion)

                    cv.imwrite(os.path.join(out_path_emotion, image), roi_color)
        else:
            continue

end = time.time()
print(end - start)

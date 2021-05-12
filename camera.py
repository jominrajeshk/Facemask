# Library Imports
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from detect import detect_and_predict_mask


class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.vs = cv2.VideoCapture(0)
        # load model
        prototxtPath = r"face_detector/deploy.prototxt"
        weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.maskNet = load_model("mask_detector.model")

    def __del__(self):
        self.vs.release()


    def get_frame(self):
        ret, frame = self.vs.read()
        frame = imutils.resize(frame, width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, self.faceNet, self.maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
      
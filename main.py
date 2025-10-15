import cv2

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    BackMask = backSub.apply(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', BackMask)
    # wget -O efficientdet.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



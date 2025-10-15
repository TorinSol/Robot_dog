import cv2

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # Default is people, but can detect faces at a stretch
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    haar_frame = frame.copy()
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(haar_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
    #
    # hog_frame = frame.copy()
    # boxes, _ = hog.detectMultiScale(hog_frame, winStride=(8, 8))
    # for (x, y, w, h) in boxes:
    #     cv2.rectangle(hog_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    #call the output "hog_frame" and itll print it
    combined = cv2.hconcat([haar_frame, hog_frame])
    cv2.imshow("Haar (Left) vs HOG+SVM (Right)", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

from tensorflow.keras.models import load_model,Sequential,Model
import cv2
import numpy as np

BOUNDING_BOX_COLOR = (0, 0, 255)
BOUNDING_BOX_THICKNESS = 4
BOUNDING_BOX_SIZE = 800

SIZE = 128

model = load_model("best_model.keras")
model.summary()

cap = cv2.VideoCapture(1)


def outputs_to_labels(outputs):
    index = np.argmax(outputs,axis=1)
    letters = np.vectorize(chr)(65 + index)
    letters[letters == ["["]] = ("----")
    return letters


ret, frame = cap.read()
shape = frame.shape
center = (int(shape[0]/2), int(shape[1]/2))
bounding_box_overlay = np.zeros((shape[0], shape[1]))
bounding_box_overlay[
    int(center[0]-BOUNDING_BOX_SIZE/2):int(center[0]+BOUNDING_BOX_SIZE/2),
    int(center[1]-BOUNDING_BOX_SIZE/2):int(center[1]+BOUNDING_BOX_SIZE/2)
    ]=np.full((BOUNDING_BOX_SIZE, BOUNDING_BOX_SIZE),1)
bounding_box_overlay[
    int(center[0]-BOUNDING_BOX_SIZE/2 + BOUNDING_BOX_THICKNESS):int(center[0]+BOUNDING_BOX_SIZE/2 - BOUNDING_BOX_THICKNESS),
    int(center[1]-BOUNDING_BOX_SIZE/2 + BOUNDING_BOX_THICKNESS):int(center[1]+BOUNDING_BOX_SIZE/2 - BOUNDING_BOX_THICKNESS)
    ]=np.zeros((BOUNDING_BOX_SIZE-2*BOUNDING_BOX_THICKNESS, BOUNDING_BOX_SIZE-2*BOUNDING_BOX_THICKNESS))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error Cant get frame")
        break
    image = frame[
        int(center[0]-BOUNDING_BOX_SIZE/2):int(center[0]+BOUNDING_BOX_SIZE/2),
        int(center[1]-BOUNDING_BOX_SIZE/2):int(center[1]+BOUNDING_BOX_SIZE/2),
        0:3].copy() / 255
    image = cv2.resize(image,(SIZE,SIZE))
    np.flip(frame,0)
    if image is not None:
        guess = model.predict(image[np.newaxis,...], verbose=0)
        print("\r",outputs_to_labels(guess)[0], end = "")
        cv2.putText(
            frame,
            str(outputs_to_labels(guess)[0]),
            (int(center[0]+BOUNDING_BOX_SIZE/2),int(center[1]-BOUNDING_BOX_SIZE/2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            BOUNDING_BOX_COLOR,
            3,
            cv2.LINE_AA
        )

    frame[bounding_box_overlay==1] = BOUNDING_BOX_COLOR
    cv2.imshow("frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# import cv2
# import numpy as np

# ESC = 27

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)

#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, binary_frame = cv2.threshold(gray_frame, 127, 256, cv2.THRESH_BINARY)

#     cv2.imshow("Camera", frame)
#     cv2.imshow("Binary", binary_frame)

#     k = cv2.waitKey(10)
#     if k == ESC & 0xff:
#         break

# cap.release()
# cv2.destroyAllWindows()

from keras.models import load_model

model = load_model('hand_gesture_recognition.h5')
model.summary()  # W celu przypomnienia.


img_path = '30.jpg'

from keras.preprocessing import image
img = image.load_img(img_path, color_mode='grayscale', target_size=(128, 128))

img_tensor = image.img_to_array(img)

import numpy as np
img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

print(img_tensor.shape)

result = model.predict(img_tensor)

prediction = {'00': result[0][0], 
                  '01': result[0][1], 
                  '02': result[0][2],
                  '03': result[0][3],
                  '04': result[0][4],
                  '05': result[0][5]}

import operator
prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

print('Result: ' + prediction[0][0])
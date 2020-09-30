import cv2
import numpy
import argparse
import tensorflow
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description = 'testing image')
parser.add_argument('--image',type = str ,help = 'location of image')
args = vars(parser.parse_args())

img = cv2.imread(args['image'],0)

''' press q to quit '''
while True :	
	cv2.imshow('image',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break;
cv2.destroyAllWindows()		

img = cv2.resize(img,(28,28))
img = numpy.expand_dims(img,axis = 0)
img = cv2.bitwise_not(img)

model = load_model('digits')

prediction = model.predict(img)

print(numpy.argmax(prediction))
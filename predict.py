
from keras.models import load_model
import h5py
from model import generator, load_samples
import itertools
import numpy as np


def main():
	print("Start")
	model = load_model('model.h5')
	print("summary", model.summary())
	samples = load_samples()
	data = next(generator(samples, batch_size=32))
	X=[]
	Y=[]
	for x,y in zip(data[0], data[1]):
		print("y:",y)
		X.append(x)
		Y.append(y)
	X = np.array(X)
	Y = np.array(Y)
	predictions = model.predict(X)
	print(len(predictions))
	print("predictions", predictions)
	for prediction,y in zip(predictions,Y):
		prediction = prediction[0]
		print("prediction:",prediction, "y",y, "diff:",prediction-y)
	score = model.evaluate(X,Y)
	print("score", score)




if __name__=='__main__':
	main()
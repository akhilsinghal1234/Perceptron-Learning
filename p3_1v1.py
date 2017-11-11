from pylab import rand,plot,show,norm
import matplotlib.pyplot as plt
import numpy
import sys
from classify_1 import arg_max,parameters

color = ['tomato','navy','hotpink','greenyellow','skyblue','khaki','lightcoral']

def test_data(files):
	inputs,k = [],0
	for f in files:
		k = k+1
		fp = open(f,"r")
		for line in fp:
			m,n = line.split()
			inputs.append([float(m),float(n),k])
		fp.close()
	return inputs

def generateData(files,i,j):
	inputs = []
	flag = -1
	fp = open(files[i],"r")
	for line in fp:
		m,n = line.split()
		inputs.append([float(m),float(n),flag])
	fp.close()
	flag = 1
	fp = open(files[j],"r")
	for line in fp:
		m,n = line.split()
		inputs.append([float(m),float(n),flag])
	fp.close()
	return inputs

def label(w,x):
	y = 0
	for i in range(len(w)):
		y += x[i] * w[i]
	return y

def find_range(testset):
	max_x,min_x,max_y,min_y = -sys.maxsize,sys.maxsize,-sys.maxsize,sys.maxsize
	for data in testset:
		if data[0] > max_x:
			max_x = data[0]
		if data[0] < min_x:
			min_x = data[0]
		if data[1] > max_y:
			max_y = data[1]
		if data[1] < min_y:
			min_y = data[1]
	return ([max_x,min_x,max_y,min_y])

class perceptron:
	def __init__(self):
		self.w = rand(3)*2 - 1
		# self.w = [-0.5,1]
		self.learningRate = 0.1
		self.pair = [0,0]

	def pairs(self,a,b):
		self.pair[0] = a
		self.pair[1] = b

	def dot(self,x):
		y = 0
		for i in range(len(self.w)):
			y += x[i] * self.w[i]
		return y

	def predicted(self,x):
		y = 0
		for i in range(len(self.w)):
			y += x[i] * self.w[i]
		if y >= 0:
			return 1				# Class +
		else:
			return -1				# Class -

	def updateWeights(self,x,iterError):
		"""
		w(t+1) = w(t) + learningRate * (d-r) * x where d is desired output, r is the perceptron 
		predicted and (d-r) is the iteration error
		"""
		for i in range(len(self.w)):
			self.w[i] +=  self.learningRate*iterError*x[i]

	def train(self,data):
		"""
		Every vector in data must have three elements, the third element (x[2]) must be the label
		"""
		learned = False	
		iteration = 0
		while not learned:
			globalError = 0
			i,j = 0,0.0
			for i in range(len(data)):
				Xn = []
				Xn = list(data[i])
				Xn.insert(0,1)
				r = self.predicted(Xn)    
				if data[i][2] != r:
					j += data[i][2] * self.dot(data[i])
					iterError = data[i][2]
					self.updateWeights(Xn,iterError)
					globalError += 1
			# print("misclassified: ",globalError, "  error fn: ",-j)
			iteration += 1
			if globalError == 0 or iteration > 100:
				print("finished")
				learned = True

def main():
	n = int(input("Number of files:"))
	files,test_files = [],[]
	for i in range (n):
		name = input("")
		files.append(name)

	for i in range (n):
		name = input("")
		test_files.append(name)
	perceptron_i,total = [],0

	for i in range(n):
		for j in range(i+1,n):
			total = total + 1
			p = perceptron()
			perceptron_i.append(p)
	k = 0
	for i in range(n):
		for j in range(i+1,n):
			trainset = []
			trainset = generateData(files,i,j)
			perceptron_i[k].train(trainset)
			perceptron_i[k].pairs(i+1,j+1)
			print(perceptron_i[k].pair)
			k += 1

	testset = test_data(test_files)
	# data = testset[185]
	# arg_max(perceptron_i,[1,data[0],data[1]])
	r = find_range(testset)
	x_r = numpy.linspace(int(r[1])-1,int(r[0])+1,200)
	y_r = numpy.linspace(int(r[3])-1,int(r[2])+1,200)
	data_x,data_y = [],[]

	for i in range(n):
		x,y = [],[]
		data_x.append(x)
		data_y.append(y)
	error_freq = 0
	for data in testset:
		k = arg_max(perceptron_i,[1,data[0],data[1]])
		if k != data[2]:
			error_freq += 1
	print("error_freq: ",error_freq)

	for x in x_r:
		for y in y_r:
			k = arg_max(perceptron_i,[1,x,y])
			data_x[k-1].append(x)
			data_y[k-1].append(y)
	for i in range(n):
		plt.scatter(data_x[i],data_y[i],color = color[i+3])

	x_test,y_test = [],[]
	str = ""
	for i in range(n):
		x,y = [],[]
		x_test.append(x)
		y_test.append(y)

	for data in testset:
		k = data[2]
		x_test[k-1].append(data[0])
		y_test[k-1].append(data[1])

	for i in range(n):
		name = test_files[i]
		name = name[5:]
		name = name[:6]
		str = str + name[5] + '_'
		plt.scatter(x_test[i],y_test[i],label = name,color=color[int(name[5])-1])
	# print(str)
	parameters(testset,perceptron_i)
	plt.legend()
	plt.savefig("fig" + str + ".png")

if __name__ == '__main__':
	main()
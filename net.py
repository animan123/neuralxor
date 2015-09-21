from numpy import *
from random import randint,seed

DEBUG=True

def debug(msg):
	if(DEBUG):
		print msg

def sigmoid(x):
	return (1.0 / (1.0 +exp(-x)))

class network:
	def __init__(self,first_layer,second_layer,third_layer):
		#Set sizes of the network layers
		self.first_layer=first_layer
		self.second_layer=second_layer
		self.third_layer=third_layer

		#defining and initializing the two weight matrices
		self.W1=empty([first_layer,second_layer])
		self.W1=random.normal(0.5,0.1,(first_layer,second_layer))
		#debug(self.W1)
		self.W2=empty([second_layer,third_layer])
		self.W2=random.normal(0.5,0.1,(second_layer,third_layer))
		#debug(self.W2)

		#defining and initializing the two biases T1 and T2
		self.T1=empty([1,second_layer])
		self.T1=random.normal(0.5,0.1,(1,second_layer))
		#debug(self.T1)
		self.T2=empty([1,third_layer])
		self.T2=random.normal(0.5,0.1,(1,third_layer))
		#debug(self.T2)

		#defining and initializing a,B,b,C,c
		self.a=empty([1,first_layer])
		self.B=empty([1,second_layer])
		self.b=empty([1,second_layer])
		self.C=empty([1,third_layer])
		self.c=empty([1,third_layer])

		#Set the learning rate
		self.learning=0.1
		
	def forward(self,input_matrix):
		if(input_matrix.size != self.a.size):
			print "Error in input"
			return
		self.a=input_matrix
		self.B=dot(self.a,self.W1) + self.T1
		self.b=sigmoid(self.B)
		self.C=dot(self.b,self.W2) + self.T2
		self.c=sigmoid(self.C)		

	def backpropogate(self,input_matrix,output_matrix):
		#Last layer
		dEdW2=empty([self.second_layer,self.third_layer])
		dEdW1=empty([self.first_layer,self.second_layer])
		delta=empty([1,self.third_layer])
		delta2=empty([1,self.second_layer])
		self.forward(input_matrix)
		print "Answer before propogation : ",self.c
		delta=(output_matrix-self.c) * (self.c) * (1-self.c)
		bdash=self.b.transpose()
		dEdW2=dot(bdash,delta)
		#print "Original : ",self.W2
		self.W2=self.W2+self.learning*dEdW2
		#print "Correction matrix : ",dEdW2
		#print "After correction : ",self.W2

		self.T2=self.T2 + self.learning*delta

		adash=self.a.transpose()
		W2dash=self.W2.transpose()
		delta2=dot(delta,W2dash) * (self.b) * (1-self.b)
		dEdW1=dot(adash,delta2)
		#print "Original : ",self.W1
		self.W1=self.W1+self.learning*dEdW1
		#print "Correction matrix : ",dEdW1
		#print "After correction : ",self.W1

		self.T1=self.T1 + self.learning * delta2
		self.forward(input_matrix)
		print "Answer after propogation : ",self.c		

N=network(2,3,1)
while True:
	print "1)Test network\n2)Train network\n3)Stop\n Enter : "
	x=int(raw_input())
	if(x>2 or x<1):
		break
	if(x==1):
		print "Enter input : "
		x,y=raw_input().split()
		x=int(x)
		y=int(y)
		N.forward(array([[x,y]]))
		print N.c
	if(x==2):
		print "Enter number of iterations : "
		n=int(raw_input())
		while(n > 0):
			seed()
			n=n-1
			x=randint(0,1)
			y=randint(0,1)
			z=int(x!=y)
			print x," ",y," ",z
			N.backpropogate(array([[x,y]]),array([[z]]))
				

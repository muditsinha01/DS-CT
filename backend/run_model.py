from Echocardiogram import *
A=Preprocessing('./Videos/')
A.preprocess()
B=EfSe(destinationFolder, videosFolder, DestinationForWeights)
if __name__ == '__main__':
	B.main()

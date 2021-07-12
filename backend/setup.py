import os
if not os.path.exists("dynamic"):
	os.system("git clone https://github.com/echonet/dynamic.git")
	os.system('py -m pip install wget')
	os.system('py -m pip install pydicom')
else :
	print("Repository  already loaded")

import os
import glob

images = glob.glob('C:\\Users\\heman\\Downloads\\task_5\\NEW\\*jpg')
c=0
for name in images:
	# print(name)
	os.rename(name,f'C:\\Users\\heman\\Downloads\\task_5\\NEW\\{str(c).zfill(5)}.jpg')
	c+=1
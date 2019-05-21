from skimage.io import imread, imsave
import os
import numpy as np
import copy
import sys
import time

start = time.time()
def img_norm(A):
    M = copy.deepcopy(A)
    M -= np.min(M)
    M /= np.max(M)
    M = (M*255).astype(np.uint8)
    return M
#picked_list = ['1.jpg','10.jpg','22.jpg','187.jpg','253.jpg'] 
IMAGE_PATH = os.path.join(os.path.abspath('.') , sys.argv[1])
#print(IMAGE_PATH)
filelist = os.listdir(IMAGE_PATH)
#print(filelist)
if '.DS_Store' in filelist:
    filelist.remove('.DS_Store')
#filelist = sorted(filelist, key=lambda x: int(x.split('.')[0]))
#print(filelist)
#input()

img_shape = imread(IMAGE_PATH+'1.jpg').shape
img_npy_name = 'img_data.npy'

if os.path.isfile(img_npy_name):
    img_data = np.load(img_npy_name)
else:
    img_data = []

    for i, filename in enumerate(filelist):
        if '.jpg' in filename and filename[:1] != '.':
            #print(filename)
            img_data.append(imread(IMAGE_PATH+filename).reshape(-1,))
    img_data = np.array(img_data)
    np.save(img_npy_name , img_data)

#print(img_data.shape)
img_data =img_data.astype('float32')
mean = np.mean(img_data, axis=0)
#x = (img_data-mean)
img_data -= mean 
x = img_data.T
del img_data
if os.path.isfile('u.npy') and os.path.isfile('s.npy')and os.path.isfile('v.npy'):
    u =np.load('u.npy')
    s = np.load('s.npy')
    v = np.load('v.npy') 
else:
    u, s, v = np.linalg.svd(x, full_matrices=False)
    np.save('u.npy', u)
    np.save('s.npy' , s)
    np.save('v.npy' , v)


eigonvector = u.T
#problem a
print("solving a")
average = img_norm(mean)
imsave('average.jpg', average.reshape(img_shape))
#problem b
print("solving b")
for x in range(5):
    eigenface = img_norm(eigonvector[x])
    imsave('eigenface_'+str(x)+'.jpg'  , eigenface.reshape(img_shape))
#problem c
print("solving c")

picked_img = imread(os.path.join(IMAGE_PATH , sys.argv[2])).flatten().astype('float32')
weight = np.array([np.dot(picked_img-mean, eigonvector[i]) for i in range(415)])
reconstruct = img_norm(np.dot(eigonvector[:5].T, weight[:5])+mean.astype(np.uint8))
imsave(sys.argv[3], reconstruct.reshape(img_shape))

#myreconstruct = img_norm(np.dot(eigonvector.T, weight)+mean.astype(np.uint8))
#imsave('myreconstruction.jpg', myreconstruct.reshape(img_shape))
#problem d
print("solving d")
p5= []
for i in range(5):
    p5.append(s[i] * 100 / sum(s))
    #print(number)
print("p5:", p5)
print("time:" , time.time()-start)


import numpy as np
from keras.models import load_model
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from keras import backend as K
import tensorflow as tf
from cutdata import loadval
from numpy.random import seed
import sys
import os
#seed(65)
from tensorflow import set_random_seed
#set_random_seed(65)
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
#global variables
model = load_model('final.h5')
class_index = [0,299,2,7,3,15,4]
  # save prefix
base_dir = './'
outdir = os.path.join(base_dir, sys.argv[2])
if not os.path.exists(outdir):
  os.mkdir(outdir)
def main():
  # load dataand model
  print("loading data...")
  x_train, y_train = loadval(sys.argv[1])
  x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
  y_train = to_categorical(y_train)

  

  layer_dict = dict([(layer.name, layer) for layer in model.layers])
  #print('Layer dict', layer_dict)
  #print(model.summary())

  y_predict = model.predict(x_train)
  y_predict = np.argmax(y_predict,axis=1)
  print(y_predict)
  
  label = [False for i in range(7)]
  #p1
  """
  for i in range(len(class_index)):
    
    #if label[y_predict[i]] == False:
    print("class ",i)
    img = x_train[class_index[i]]
    mask = get_Saliency(model, img, y_predict[class_index[i]])

    plot_img(img.reshape((48, 48)), 'gray' ,i)
    plot_img(mask.reshape((48, 48)), 'jet' , i)
    #label[y_predict[i]] = True
"""
  #p2
  vis_img_in_filter( img=np.zeros((1, 48, 48, 1)).astype(np.float64), layer_dict=layer_dict , outname='fig2_1.png')
  vis_img_in_filter( img=x_train[0].reshape((1, 48, 48, 1)).astype(np.float64), layer_dict=layer_dict , outname='fig2_2.png')
  #p3
  print("using lime...")
  seed(66)
  set_random_seed(66)

  explainer = lime_image.LimeImageExplainer()
  #prediction: 4 0 4 6 2 5 6(only last one correct)
  for i in range(len(class_index)):
    #im = plt.imshow(x_train[class_index[i]].reshape((48, 48)) , cmap='gray')
    #plt.colorbar(im)
    #plt.show()
    #input()
    print("plotting class "+ str(i))
    explanation = explainer.explain_instance(x_train[class_index[i]].reshape((48,48)), classifier_fn=my_predict, top_labels=5, hide_color=0, num_samples=1000, segmentation_fn=my_slic, random_seed=66)

    temp, mask = explanation.get_image_and_mask(y_predict[class_index[i]], positive_only=True, num_features=5, hide_rest=False)
    fig = plt.figure()
    ax = plt.axes()
    ax.imshow(mark_boundaries(temp, mask))
    print("predict class:" , y_predict[class_index[i]])
    ax.grid('off')
    ax.axis('off')
    #plt.show()
    fig.savefig(os.path.join(outdir, 'fig3_'+str(i)+'.png'))
    #plt.imsave(os.path.join(outdir, 'fig3_'+str(i)+'.png') , temp)
    
#print(model.summary())

# Calculate Saliency Map
def get_Saliency(model, input_image, output_index = 0):
    # Define the function to compute the gradient
    input_tensors = [model.input]
    gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
    compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    # Execute the function to compute the gradient
    x_value = np.expand_dims(input_image, axis=0)
    gradients = compute_gradients([x_value])[0][0]

    return gradients

def plot_img(img, cm , i):
  fig = plt.figure()
  ax = plt.axes()
  ax.imshow(img, cmap=cm)
  ax.grid('off')
  ax.axis('off')
  fig.savefig(os.path.join(outdir, 'fig1_'+str(i)+'.png'))
  #plt.show()

def add_mask(img, mask):
  masked = np.zeros_like(img)
  avg = np.average(mask)
  for i in range(48*48):
    if mask[i] > avg:
      masked[i] = img[i]
  return masked



# visualize layer
# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def vis_img_in_filter(img,outname , layer_dict ,   layer_name = 'zero_padding2d_2' ):
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    #for filter_index in range(layer_output.shape[3]):
    for filter_index in range(64):
        print("processing filter ", filter_index, " ......")
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 1.

        img_asc = np.array(img)
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))
        
    plot_x, plot_y = 11, 6
    print("plotting......")
    #fig = plt.figure()
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (32, 32))
    ax[0, 0].imshow(img.reshape((48, 48)), cmap = 'gray')
    ax[0, 0].set_title('Input image')
    fig.suptitle('Input image and %s filters' % (layer_name,) , fontsize = 32)
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        if x == 0 and y == 0 or x * plot_y + y - 1 > 63 :
            continue
        ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
        ax[x, y].grid('off')
        ax[x, y].axis('off')
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1) , fontsize = 18)
    fig.savefig(os.path.join(outdir,outname))





# https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb


def my_predict(imgs):
  gray_imgs = imgs[:,:,:,0:1]
  return(model.predict(gray_imgs))

def my_slic(img):
  return slic(img, n_segments=100)



if __name__ == "__main__":
  main()
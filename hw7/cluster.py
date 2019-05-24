
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.cluster import KMeans
from keras.preprocessing.image import ImageDataGenerator
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
import argparse
from skimage.io import imread, imsave
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

is_gray=False



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_images_imread(data_dir,isgray=False):
    images = []
    image_list = os.listdir(data_dir)
    image_list.sort()
    #print(image_list)
    for i in image_list:
        if 'jpg' in i:
            images.append(imread(os.path.join(data_dir , i), as_gray=isgray) )
            #print(imread(os.path.join(data_dir , i), as_gray=True))
            #input()
    images = np.array(images)
    
    np.save(f"images_imread_isgray_{int(isgray)}.npy", images)
    return images
def load_images_PIL(data_dir,isgray=False):
    images = []
    image_list = os.listdir(data_dir)
    image_list.sort()
    #print(image_list)
    #input()
    #print(image_list)
    for i in image_list:
        if 'jpg' in i:
            if is_gray:
                img=Image.open(os.path.join(data_dir , i)).convert('LA')
            else:
                img=Image.open(os.path.join(data_dir , i))
            data = np.array( img )
            images.append(data)
    images = np.array(images)
    
    np.save(f"images_PIL_isgray_{int(isgray)}.npy", images)
    return images
def main(args):

    # build model
    MODEL_FILE = args.encoder
    AUTOENCODE_MODEL_FILE=args.autoencoder
    
    input_img = Input(shape=(32,32,3 ))
    x = Conv2D(32, (3, 3), activation='relu', padding='same',data_format="channels_last")(input_img)
    x = MaxPooling2D((2, 2), padding='same',data_format="channels_last")(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',data_format="channels_last")(x)
    #x = MaxPooling2D((2, 2), padding='same',data_format="channels_last")(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',data_format="channels_last")(x)
    #x = Flatten()(x)
    #encoded = Dense(512, activation='relu', padding='same',data_format="channels_last")(x)
    encoded = MaxPooling2D((2, 2), padding='same',data_format="channels_last")(x)
     
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    #x = Reshape(())
    x = Conv2D(16, (3, 3), activation='relu', padding='same',data_format="channels_last")(encoded)
    #x = UpSampling2D((2, 2),data_format="channels_last")(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',data_format="channels_last")(x)
    x = UpSampling2D((2, 2),data_format="channels_last")(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',data_format="channels_last")(x)
    x = UpSampling2D((2, 2),data_format="channels_last")(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same',data_format="channels_last")(x)
    
    """
    input_img = Input(shape=(32*32,))
    encoded = Dense(1024, activation='relu')(input_img)
    encoded = Dense(512, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)

    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(32*32, activation='linear')(decoded)
"""
    # build encoder
    encoder = Model(input=input_img, output=encoded)

    # build autoencoder

    adam = Adam(lr=5e-4)
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.summary()

    # load images
    if os.path.isfile(f"images_PIL_isgray_{int(is_gray)}.npy"):
        X = np.load(f"images_PIL_isgray_{int(is_gray)}.npy")
    else:
        X = load_images_PIL(args.data_dir,isgray=is_gray)
    train_num = int(X.shape[0]*0.9)
    """
    for i in range(X.shape[0]):
        print(X[i])
        input()
    """
    #X = rgb2gray(X)
    #X = X.astype('float32') / 255.
    print(X.shape)
  
    #X = np.reshape(X, (len(X), -1))
    #x_train = X[:train_num,:]
    #x_val = X[train_num: ,: ]
    #print(X[0].dtype)

    if is_gray:
        X = np.expand_dims(X, axis=3)
    else:
        X = (X.astype('float32') / 255.)
    x_train = X[:train_num,:,:]
    x_val = X[train_num: ,: ,:]
    print(x_train.shape, x_val.shape)
    """
    x0 = (X[0]*255).astype('uint8')
    print(x0)
    imsave('trainx[0].png', x0)
    plt.imshow(x0)
    plt.savefig("trainx[0]_plt.png")
    """
    #print(x_train[1])
    #input()

    # train autoencoder

    if os.path.isfile(MODEL_FILE):
        encoder = load_model(MODEL_FILE)
        autoencoder = load_model(AUTOENCODE_MODEL_FILE)

    else:
        callbacks = [EarlyStopping('val_loss', patience=5), 
                    ModelCheckpoint(MODEL_FILE, save_best_only=True)]
        autoencoder.fit(x_train, x_train, epochs=300, batch_size=128, shuffle=True, validation_data=(x_val, x_val), callbacks=callbacks)
        encoder.save(MODEL_FILE)
        autoencoder.save(AUTOENCODE_MODEL_FILE)
        

    # after training, use encoder to encode image, and feed it into Kmeans
    encoded_imgs = encoder.predict(X)
    encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)

    #if args.tsne != 0:
     #   cluster_imgs = TSNE(n_components=2, n_iter = 3000, verbose=1).fit_transform(encoded_imgs)
    if args.pca != 0 :
        pca=PCA(n_components=args.pca,whiten=True,random_state=0)
        cluster_imgs=pca.fit_transform(encoded_imgs)
        
       
    #print(pca_imgs)
    #input()
    kmeans = KMeans(n_clusters=2,max_iter=4000, random_state=0, n_jobs=4).fit(cluster_imgs)
    #print(kmeans.labels_)
    #input()

    # get test cases
    f = pd.read_csv(args.test_x) # test_case.csv
    IDs, idx1, idx2 = np.array(f['id']), np.array(f['image1_name']), np.array(f['image2_name'])
    #print(idx1)
    #input()
    # predict
    o = open(args.output, 'w') # prediction file path
    o.write("id,label\n")
    for idx, i1, i2 in zip(IDs, idx1, idx2):
        #i1 = str(int(i1))
        #i2 = str(int(i2))

        p1 = kmeans.labels_[int(i1)-1]
        p2 = kmeans.labels_[int(i2)-1]
        #print(i1,':',p1 , i2 ,':', p2)
        if p1 == p2:
            pred = 1  # two images in same cluster
        else: 
            pred = 0  # two images not in same cluster
        o.write("{},{}\n".format(idx, pred))
    o.close()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str , help='[Input] Your image dir')
    parser.add_argument('test_x',type=str, help='[Input] Your test_x.csv')
    parser.add_argument('output',type=str, help='[Output] name of prediction csv')
    parser.add_argument('--encoder',  default='encoder.h5' ,type=str)
    parser.add_argument('--autoencoder',  default='autoencoder.h5' ,type=str)
    parser.add_argument('--pca',  default=700 ,type=int)  
    parser.add_argument('--tsne',  default=0 ,type=int)   

    
    args = parser.parse_args()
    main(args)
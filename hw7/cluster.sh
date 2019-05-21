wget https://www.dropbox.com/s/sypl7v0gfevehgb/autoencoder_model_CNN_nogray_0.94811.h5?dl=1 -O autoencoder.h5
wget https://www.dropbox.com/s/gn1oqmbultmwzho/encoder_model_CNN_nogray_0.94811.h5?dl=1 -O encoder.h5
python cluster.py $1 $2 $3

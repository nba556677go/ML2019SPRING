wget https://www.dropbox.com/s/hi9675fa6ijk8oy/nn-epoch-30-ensemble5.pt?dl=1 -O 5.pt
wget https://www.dropbox.com/s/tqavz9s1cotzmp8/nn-epoch-33-ensemble4.pt?dl=1 -O 4.pt
wget https://www.dropbox.com/s/nv4ef6dcg3j06n8/nn-epoch-40-ensemble3.pt?dl=1 -O 3.pt
wget https://www.dropbox.com/s/fa1nf1k1hlmw199/nn-epoch-44-ensemble2.pt?dl=1 -O 2.pt
wget https://www.dropbox.com/s/fnn1t1wszt2xik7/nn-epoch-46-ensemble1.pt?dl=1 -O 1.pt
python hw3_test.py $1 $2

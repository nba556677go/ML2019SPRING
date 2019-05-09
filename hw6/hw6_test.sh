wget https://www.dropbox.com/s/xk9b4nna9f2317m/ensemble0lstm_shieh_75.722_seq40_dim256_numlayer2?dl=1 -O models/1.pt
wget https://www.dropbox.com/s/mto95oofwjbdmnr/ensemble1lstm_shieh_75.974_seq40_dim256_numlayer2?dl=1 -O models/2.pt
wget https://www.dropbox.com/s/5vtuzwqpkxcfogl/ensemble2lstm_shieh_75.748_seq40_dim256_numlayer2?dl=1 -O models/3.pt
wget https://www.dropbox.com/s/c8fpwuaaxmvm73f/ensemble3lstm_shieh_75.907_seq40_dim256_numlayer2?dl=1 -O models/4.pt
wget https://www.dropbox.com/s/0bhnioinbdraig5/ensemble4lstm_shieh_75.588_seq40_dim256_numlayer2?dl=1 -O models/5.pt
wget https://www.dropbox.com/s/4kz559tgssc6lsp/ensemble5lstm_shieh_75.336_seq40_dim256_numlayer2_epoch1?dl=1 -O models/6.pt
wget https://www.dropbox.com/s/xwp78ngp16y3ob1/ensemble6lstm_shieh_76.050_seq40_dim256_numlayer2_epoch3?dl=1 -O models/7.pt
wget https://www.dropbox.com/s/h7juuex0t54502b/ensemble7lstm_shieh_75.521_seq40_dim256_numlayer2_epoch2?dl=1 -O models/8.pt
wget https://www.dropbox.com/s/n4i5plhyt1g5edf/ensemble8lstm_shieh_75.227_seq40_dim256_numlayer2_epoch2?dl=1 -O models/9.pt
wget https://www.dropbox.com/s/cn6b736yz3hu74a/ensemble9lstm_shieh_76.126_seq40_dim256_numlayer2_epoch2?dl=1 -O models/10.pt
wget https://www.dropbox.com/s/3yz2xrmcoar6q6k/word2vec?dl=1 -O word2vec
wget https://www.dropbox.com/s/y49a5sv1j5becms/word2vec.trainables.syn1neg.npy?dl=1 -O word2vec.trainables.syn1neg.npy
wget https://www.dropbox.com/s/na0sp2m16otzf0b/word2vec.wv.vectors.npy?dl=1 -O word2vec.wv.vectors.npy
python hw6_test.py $1 $2 $3
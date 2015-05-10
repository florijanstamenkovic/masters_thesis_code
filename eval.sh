sudo THEANO_FLAGS='device=gpu,force_device=True,cuda.root=/usr/local/cuda,warn_float64=raise' python nnet_rbm.py -s 2 -o 5 -f 1 -l -eps 0.1 -ep 10 -d 150 -mnb 500 -a 0.6
sudo THEANO_FLAGS='device=gpu,force_device=True,cuda.root=/usr/local/cuda,warn_float64=raise' python nnet_rbm.py -s 2 -o 5 -f 1 -l -eps 0.1 -ep 10 -d 150 -mnb 500 -a 0.4


THEANO_FLAGS='device=gpu,force_device=True,cuda.root=/usr/local/cuda,warn_float64=raise' python nnet_rbm.py -s 4 -o 5 -f 1 -eps 0.005 -ep 2 -d 100 -mnb 500 -a 0.6
THEANO_FLAGS='device=gpu,force_device=True,cuda.root=/usr/local/cuda,warn_float64=raise' python nnet_rbm.py -s 4 -o 5 -f 1 -eps 0.002 -ep 2 -d 100 -mnb 500 -a 0.6


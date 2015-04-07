cd ..
echo "Evaluating ngram models, batch 2"
python ngram.py -o 5 -f 3 -u 100000 -e -t
python ngram.py -o 5 -f 3 -u 010000 -e -t
python ngram.py -o 5 -f 3 -u 010001 -e -t
python ngram.py -o 5 -f 3 -u 001000 -e -t
echo "Done evaluating ngram models, batch 2"

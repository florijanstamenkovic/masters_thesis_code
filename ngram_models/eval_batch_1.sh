cd ..
echo "Evaluating ngram models, batch 1"
python ngram.py -o 5 -f 3 -u 100000 -e
python ngram.py -o 5 -f 3 -u 010000 -e
python ngram.py -o 5 -f 3 -u 010100 -e
python ngram.py -o 5 -f 3 -u 001000 -e
echo "Done evaluating ngram models, batch 1"

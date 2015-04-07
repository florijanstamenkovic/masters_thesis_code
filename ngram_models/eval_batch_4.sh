cd ..
echo "Evaluating ngram models, batch 4"
python ngram.py -o 5 -f 3 -u 010100 -t -e
python ngram.py -o 5 -f 3 -u 010101 -t -e
python ngram.py -o 5 -f 3 -u 001100 -t -e
python ngram.py -o 5 -f 3 -u 001101 -t -e
echo "Done evaluating ngram models, batch 4"

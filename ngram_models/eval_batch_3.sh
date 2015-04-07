cd ..
echo "Evaluating ngram models, batch 3"
python ngram.py -o 5 -f 3 -u 100100 -e
python ngram.py -o 5 -f 3 -u 001100 -e
python ngram.py -o 5 -f 3 -u 100001 -t -e
python ngram.py -o 5 -f 3 -u 001001 -t -e
echo "Done evaluating ngram models, batch 3"
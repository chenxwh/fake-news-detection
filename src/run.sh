#!/bin/sh

#python main.py --mode train --model ernie --logging --dataset liar --threshold 0.1
#python main.py --mode test --model ernie --logging --dataset liar --threshold 0.1
#python main.py --mode train --model ernie --logging --dataset liar --threshold 0.2
#python main.py --mode test --model ernie --logging --dataset liar --threshold 0.2
#python main.py --mode train --model ernie --logging --dataset liar --threshold 0.4
#python main.py --mode test --model ernie --logging --dataset liar --threshold 0.4
#python main.py --mode train --model ernie --logging --dataset liar --threshold 0.5
#python main.py --mode test --model ernie --logging --dataset liar --threshold 0.5
#python main.py --mode train --model knowbert-w-w --logging
#python main.py --mode test --model knowbert-w-w --logging
#python main.py --mode train --model knowbert-w-w --logging --dataset liar
#python main.py --mode test --model knowbert-w-w --logging --dataset liar
python main.py --mode train --model roberta-large --logging --dataset liar
python main.py --mode test --model roberta-large --logging --dataset liar
python main.py --mode train --model ernie --logging --dataset liar --threshold 0.1 --num_labels 2
python main.py --mode test --model ernie --logging --dataset liar --threshold 0.1 --num_labels 2
python main.py --mode train --model ernie --logging --dataset liar --threshold 0.2 --num_labels 2
python main.py --mode test --model ernie --logging --dataset liar --threshold 0.2 --num_labels 2
python main.py --mode train --model ernie --logging --dataset liar --threshold 0.4 --num_labels 2
python main.py --mode test --model ernie --logging --dataset liar --threshold 0.4 --num_labels 2
python main.py --mode train --model ernie --logging --dataset liar --threshold 0.5 --num_labels 2
python main.py --mode test --model ernie --logging --dataset liar --threshold 0.5 --num_labels 2
python main.py --mode train --model knowbert-w-w --logging --dataset liar --num_labels 2
python main.py --mode test --model knowbert-w-w --logging --dataset liar --num_labels 2
# Fake News Detection

Official implementation for experiments in the paper "[Evaluation of Fake News Detection with Knowledge-Enhanced Language Models](https://ojs.aaai.org/index.php/ICWSM/article/view/19400)". \
In Proceedings of the Sixteenth International AAAI Conference on Web and Social Media (AAAI ICWSM-2022).

Fake news detection with BERT, RoBERTa and various knowledge-enhanced PLMs including [ERNIE](https://arxiv.org/abs/1905.07129), [KnowBert](https://arxiv.org/abs/1909.04164), [KEPLER](https://arxiv.org/abs/1911.06136) and [K-ADAPTER](https://arxiv.org/abs/2002.01808).\
Experimented on [LIAR](https://arxiv.org/abs/1705.00648) and [COVID-19](https://arxiv.org/abs/2011.03327) dataset.

## Get Started
```
git clone https://github.com/chenxwh/fake-news-detection.git
cd fake_news_detection
```

## Install Dependencies
```
bash install_libs
```

## Download PLMs weights 
To train or test on [ERNIE](https://github.com/thunlp/ERNIE), [KnowBert](https://github.com/allenai/kb), [KEPLER](https://github.com/THU-KEG/KEPLER) and [K-ADAPTER](https://github.com/microsoft/k-adapter), we need to download the pretrained weights from the corresponding repositories.\
After downloading the wieghts, change the path to the weights in `src/config.yaml`.

## Train and Test 
Modify hyper-parameters in `src/config.yaml`.
Run following to train and rest on the fake news detection datasets
```
cd src
python main.py --mode train --dataset liar --model bert-base  --num_labels 6 --logging --verbose
python main.py --mode test --dataset liar --model bert-base --num_labels 6 --logging
```

## Citation
if you find the work helpful, please consider citing:
```
@inproceedings{whitehouse2022evaluation,
  title={Evaluation of Fake News Detection with Knowledge-Enhanced Language Models},
  author={Whitehouse, Chenxi and Weyde, Tillman and Madhyastha, Pranava and Komninos, Nikos},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={16},
  pages={1425--1429},
  year={2022}
}
```

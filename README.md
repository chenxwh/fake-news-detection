# Fake News Detection

Fake news detection with BERT, RoBERTa and various knowledge-enhanced PLMs\
Experimented on Liar and COVID-19 dataset.

Download the fine-tuned PLMs [here](https://drive.google.com/drive/folders/1E-PwWR1P_OOP6FK_IeKCIh-jy7mWZcgs?usp=sharing)

Install dependencies
```
pip install torch
pip install torchvision
pip install -r requirements.txt
```

Modify hyper-parameters in `src/config/yaml`.
Run following to train and rest on the fake news detection datasets
```
cd src
python main.py --mode train --dataset covid --model bert-base
python main.py --mode test --dataset covid --model bert-base
```


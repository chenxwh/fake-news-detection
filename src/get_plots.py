import numpy as np
import matplotlib.pyplot as plt

labels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '1']
test = [25688, 14000, 8318, 5798, 3901, 685, 0]
val = [24890, 13762, 8208, 5800, 3958, 715, 0]
train = [74566, 40896, 24373, 17029, 11500, 2019, 0]

x = np.arange(len(labels))  # the label locations
width = 0.3 # the width of the bars

fig, ax = plt.subplots(figsize=(12,6))
rects1 = ax.bar(x - width, train, width, label='Train')
rects2 = ax.bar(x , val, width, label='Validation')
rects3 = ax.bar(x + width, test, width, label='Test')

ax.set_ylabel('Linked Entities by ERNIE in COVID-19')
ax.set_xlabel('TAGME Threshold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=2)
ax.bar_label(rects2, padding=2)
ax.bar_label(rects3, padding=2)

fig.tight_layout()

plt.show()
fig.savefig('ernie_entitites_covid_thr.png', dpi=1200)
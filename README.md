# Dtap: disease trend analysis platform
The following is a detailed introduction to the main functions of the Dtap model, its parameters, and the methods it calls.
## Install
Git is recommended for Dtap installation.
```python
# install by github
pip install git+https://github.com/KingoftheNight/Dtap.git
# install by gitee
`pip install git+https://gitee.com/KingoftheNight/Dtap.git`
```
## Import Dtap
```python
import pandas as pd
from Dtap import Dtap
```
## Import Datasets
We provided two small data sets for model testing, and in the label column, 1 represents cervical cancer and 0 represents patients with other diseases.
```python
train_data = pd.read_csv('scc-train.csv')
test_data = pd.read_csv('scc-test.csv')
```
![mark](http://img.frankgene.top/blog/20230412/p76FEwytMg9n.png)
## Load Dtap
```python
classfier = Dtap()
```
## Determine The k Threshold
In Dtap, k is the most important hyperparameter. The `thresholdSearch` function is used to optimize the input dataset's hyperparameter and obtain the Dtap model with the best k threshold.
```python
classfier.thresholdSearch(Train, Test=False, indep=False, res=0, order='mcc', standard=None, model='RF')
```










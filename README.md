# Dtap: disease trend analysis platform
The following is a detailed introduction to the main functions of the Dtap model, its parameters, and the methods it calls.
## 1. Install
Git is recommended for Dtap installation.
```python
# install by github
pip install git+https://github.com/KingoftheNight/Dtap.git
# install by gitee
pip install git+https://gitee.com/KingoftheNight/Dtap.git
```
## 2. Import Dtap
```python
import pandas as pd
from Dtap import Dtap
```
## 3. Import Datasets
We provided two small data sets for model testing, and in the label column, 1 represents cervical cancer and 0 represents patients with other diseases.
```python
train_data = pd.read_csv('scc-train.csv')
test_data = pd.read_csv('scc-test.csv')
```
![mark](http://img.frankgene.top/blog/20230412/p76FEwytMg9n.png)
## 4. Load Dtap
```python
classfier = Dtap()
```
## 5. Determine The k Threshold
In Dtap, k is the most important hyperparameter. The `thresholdSearch` function is used to optimize the input dataset's hyperparameter and obtain the Dtap model with the best k threshold.
```python
classfier.thresholdSearch(value, Test=False, indep=False, res=0, order='mcc', standard=None, model='RF')
```
- `value`（Required）：DataFrame format, represents the input training set.
- `Test`（Optional）：DataFrame format, represents the input testing set, no input by default.
- `indep`（Optional）：DataFrame format, represents the input independent set, no input by default.
- `res`（Optional）：Integer format, indicates which result is selected for k threshold search, and the default value is 0 (that is, the search is conducted using Train's 5 fold cross verification result); You can change this value to 1 when you enter Test (that is, search with test set predictions); When you enter both Test and indep, you can change the value to 2 (that is, search with independent data set predictions).
- `order`（Optional）：String format, indicates the evaluation standard. The default value is 'mcc'. The optional value can be 'acc', 'sn', 'sp', 'ppv', 'mcc', 'f1'.
- `standard`（Optional）：Dict format, represents the range of standard adult blood indicators. Dtap has a built-in range of blood routine test indicators. If users want to analyze other indicators, they need to use this parameter to replace it. The standard format is as follows: {' WBC ': [3.5, 9.5]}.
- `model`（Optional）：String format, indicates the evaluation model. The default value is 'RF'. The optional value can be 'SVM', 'KNN', 'LR', 'DT', 'RF', 'XGB'.
## 6. Training
```python
classfier.fit(value, standard=None, threshold=None, model='SVM', c=None, g=None, is_trend=True)
```








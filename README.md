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
- `value`（Required）：DataFrame format, represents the input training set.
- `standard`（Optional）：Dict format, it has the same function as the parameter in part 5.
- `threshold`（Optional）：Float format, represents the k threshold. The default value is 0.2. If you search for the best results for the k threshold using the functions in part 5, you do not need to manually configure it.
- `model`（Optional）：String format, it has the same function as the parameter in part 5.
- `c`（Optional）：Integer format, represents the hyperparameter c of SVM, and the default value is None (Dtap will actively perform SVM hyperparameter optimization). If parameters c and g are given at the same time, SVM hyperparameter optimization is skipped and the specified value is directly used for training.
- `g`（Optional）：Integer format, represents the hyperparameter g of SVM, and the default value is None (Dtap will actively perform SVM hyperparameter optimization). If parameters c and g are given at the same time, SVM hyperparameter optimization is skipped and the specified value is directly used for training.
- `is_trend`（Optional）：Bool format, indicates whether to extract trend features. The default value is True. If set to False, the original features are directly used for modeling.
## 7. Predicting
```python
classfier.predict(value, threshold=0.5, is_trend=True, weight=0.024, model=None)
```
- `value`（Required）：DataFrame format, represents the input testing set.
- `threshold`（Optional）：Float format, represents the classification threshold. The default value is 0.5.
- `is_trend`（Optional）：Bool format, it has the same function as the parameter in part 5, and please ensure this parameter is consistent with the training.
- `weight`（Optional）：Float format, represents the category weight when multiple classifiers are stacked. The default value is 0.024. This parameter takes effect only when model is 'STK'.
- `model`（Optional）：Model format, represents prediction using a user-specified model. The default value is None. When you want to customize a model for prediction, make sure that the model is a model trained and saved using Dtap. Otherwise, an error may occur.
## 8. Cross Validation
```python
classfier.crossv(cv=5)
```
- `cv`（Optional）：Integer format, represents the number of folds for cross validation. The default is 5.
## 9. Selection
```python
classfier.select(rule='ANOVA', res='f1')
```
- `rule`（Optional）：String format, represents feature importance algorithm. The default value is 'ANOVA'. The optional value can be 'ANOVA', 'Correlation', 'TURF', 'PCA-SVD', 'RF-Weight', 'Lasso'.
- `res`（Optional）：String format, indicates the evaluation standard. The default value is 'f1'. The optional value can be 'acc', 'sn', 'sp', 'ppv', 'mcc', 'f1'.
## 10. Shap
```python
classfier.shap(item=0, plot=['bar', 'beeswarm', 'heatmap', 'force'], out=os.getcwd())
```
- `item`（Optional）：Integer format, represents shap values for analyzing data with the specified subscript. The default is 0.
- `plot`（Optional）：List format, indicates the type of picture to be drawn. The default value is ['bar', 'beeswarm', 'heatmap', 'force'], i.e. all drawn.
- `out`（Optional）：String format, indicates the output location. By default, the output is in the current path.
## 11. ROC Curve
```python
classfier.roc(label='ROC', color=['#525288','#F17666','#624941','#B90000','#D7A100','#1A6840','#BFAEA4'], lw=1, figsize=[5,5], dpi=300, xlabel='False Positive Rate', ylabel='True Positive Rate', legend="lower right", title='ROC_curve', out=os.getcwd())
```
- `label`（Optional）：List format, represents the label of ROC curve, default is 'ROC'.
- `color`（Optional）：List format, represents the color of ROC curves, default is ['#525288','#F17666','#624941','#B90000','#D7A100','#1A6840','#BFAEA4']. When multiple ROC curve data are entered, colors are assigned in sequence according to color.
- `lw`（Optional）：Integer format, indicates the line width of ROC curve, default is 1.
- `figsize`（Optional）：List format, represents the size of ROC curves, default is [5,5].
- `dpi`（Optional）：Integer format, indicates the image resolution of ROC curve, default is 300.
- `xlabel`（Optional）：String format, indicates the x-axis label, default is 'False Positive Rate'.
- `ylabel`（Optional）：String format, indicates the y-axis label, default is 'True Positive Rate'.
- `legend`（Optional）：String format, indicates the legend location, default is 'lower right'.
- `title`（Optional）：String format, indicates the title of ROC curve, default is 'ROC_curve'.
- `out`（Optional）：String format, it has the same function as the parameter in part 10.
## 12. PRC Curve
```python
classfier.roc(label='PRC', color=['#525288','#F17666','#624941','#B90000','#D7A100','#1A6840','#BFAEA4'], lw=1, figsize=[5,5], dpi=300, xlabel='Precision', ylabel='Recall', legend="lower left", title='PRC_curve', out=os.getcwd())
```
- `label`（Optional）：List format, it has the same function as the parameter in part 11.
- `color`（Optional）：List format, it has the same function as the parameter in part 11.
- `lw`（Optional）：Integer format, it has the same function as the parameter in part 11.
- `figsize`（Optional）：List format, it has the same function as the parameter in part 11.
- `dpi`（Optional）：Integer format, it has the same function as the parameter in part 11.
- `xlabel`（Optional）：String format, it has the same function as the parameter in part 11.
- `ylabel`（Optional）：String format, it has the same function as the parameter in part 11.
- `legend`（Optional）：String format, it has the same function as the parameter in part 11.
- `title`（Optional）：String format, it has the same function as the parameter in part 11.
- `out`（Optional）：String format, it has the same function as the parameter in part 11.
## 13. Model Evaluation
```python
classfier.evaluate(value, Test=False, indep=False, standard=None, threshold=None, models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, cv=5, is_trend=True)
```
- `value`（Required）：DataFrame format, it has the same function as the parameter in part 5.
- `Test`（Optional）：DataFrame format, it has the same function as the parameter in part 5.
- `indep`（Optional）：DataFrame format, it has the same function as the parameter in part 5.
- `standard`（Optional）：Dict format, it has the same function as the parameter in part 5.
- `threshold`（Optional）：Float format, it has the same function as the parameter in part 6.
- `models`（Optional）：String format, it has the same function as the parameter in part 10.
- `c`（Optional）：Float format, it has the same function as the parameter in part 6.
- `g`（Optional）：Float format, it has the same function as the parameter in part 6.
- `cv`（Optional）：Integer format, it has the same function as the parameter in part 8.
- `is_trend`（Optional）：Bool format, it has the same function as the parameter in part 6.

















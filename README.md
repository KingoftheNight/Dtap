# Dtap: disease trend analysis platform
The following is a detailed introduction to the main functions of the Dtap model, its parameters, and the methods it calls.
## 1. Install
Git is recommended for Dtap installation.
```bash
# install git
conda install git
```
```python
# install by github
pip install git+https://github.com/KingoftheNight/Dtap.git
# install by gitee
pip install git+https://gitee.com/KingoftheNight/Dtap.git
```
## 2. Import Dtap
```python
import pandas as pd
from dtap.Dtap import Dtap
```
## 3. Import Datasets
We provided three small data sets for model testing, and in the label column, 1 represents cervical cancer and 0 represents patients with other diseases. These data are for demonstration only, and the calculated results do not represent the real performance of the model.
```python
Train = pd.read_csv('Train-20.csv')
Test = pd.read_csv('Test-20.csv')
Indpt = pd.read_csv('Indpt-20.csv')
```
![mark](http://img.frankgene.top/blog/20230413/RdT9p1RN4uFg.png)

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
### Example
```python
classfier.thresholdSearch(Train, Test=Test, res=1, order='f1')
```
![mark](http://img.frankgene.top/blog/20230413/SRJEUz2tE0gs.png)
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
### Example
```python
classfier.fit(Train, model='RF')
classfier.model
```
![mark](http://img.frankgene.top/blog/20230413/X4lHVaxQjyxd.png)
## 7. Predicting
```python
classfier.predict(value, threshold=0.5, is_trend=True, weight=0.024, model=None)
```
- `value`（Required）：DataFrame format, represents the input testing set.
- `threshold`（Optional）：Float format, represents the classification threshold. The default value is 0.5.
- `is_trend`（Optional）：Bool format, it has the same function as the parameter in part 5, and please ensure this parameter is consistent with the training.
- `weight`（Optional）：Float format, represents the category weight when multiple classifiers are stacked. The default value is 0.024. This parameter takes effect only when model is 'STK'.
- `model`（Optional）：Model format, represents prediction using a user-specified model. The default value is None. When you want to customize a model for prediction, make sure that the model is a model trained and saved using Dtap. Otherwise, an error may occur.
### Example
```python
classfier.predict(Test)
classfier.predict_result
classfier.predict_value
```
![mark](http://img.frankgene.top/blog/20230413/YS9EXdIwYcwH.png)

![mark](http://img.frankgene.top/blog/20230413/R8znt01UOkbB.png)
## 8. Cross Validation
```python
classfier.crossv(cv=5)
```
- `cv`（Optional）：Integer format, represents the number of folds for cross validation. The default is 5.
### Example
```python
classfier.crossv(cv=5)
classfier.cv_result
```
![mark](http://img.frankgene.top/blog/20230413/xboADRdlMUdv.png)
## 9. Selection
```python
classfier.select(rule='ANOVA', res='f1')
```
- `rule`（Optional）：String format, represents feature importance algorithm. The default value is 'ANOVA'. The optional value can be 'ANOVA', 'Correlation', 'TURF', 'PCA-SVD', 'RF-Weight', 'Lasso'.
- `res`（Optional）：String format, indicates the evaluation standard. The default value is 'f1'. The optional value can be 'acc', 'sn', 'sp', 'ppv', 'mcc', 'f1'.
### Example
```python
classfier.select(rule='ANOVA', res='f1')
classfier.selection, classfier.selection.max_fs, classfier.selection.max_fv
```
![mark](http://img.frankgene.top/blog/20230413/uI4xoE4F84he.png)
## 10. Shap
```python
classfier.shap(item=0, plot=['bar', 'beeswarm', 'heatmap', 'force'], out=os.getcwd())
```
- `item`（Optional）：Integer format, represents shap values for analyzing data with the specified subscript. The default is 0.
- `plot`（Optional）：List format, indicates the type of picture to be drawn. The default value is ['bar', 'beeswarm', 'heatmap', 'force'], i.e. all drawn.
- `out`（Optional）：String format, indicates the output location. By default, the output is in the current path.
### Example
```python
classfier.shap()
classfier.shap_values
```
![mark](http://img.frankgene.top/blog/20230413/XcLyQIdyy6SU.png)

![mark](http://img.frankgene.top/blog/20230413/lAogVXDg7mxl.png)

![mark](http://img.frankgene.top/blog/20230413/MtxhioDWd4Hx.png)
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
### Example
```python
classfier.roc()
classfier.fpr, classfier.tpr
```
![mark](http://img.frankgene.top/blog/20230413/eEdCnalP3zNo.png)
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
### Example
```python
classfier.prc()
classfier.pre, classfier.rec
```
![mark](http://img.frankgene.top/blog/20230413/TFylbPjJ0BIj.png)
## 13. Models Evaluation
```python
classfier.evaluate(value, Test=False, indep=False, standard=None, threshold=None, models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, cv=5, is_trend=True)
```
- `value`（Required）：DataFrame format, it has the same function as the parameter in part 5.
- `Test`（Optional）：DataFrame format, it has the same function as the parameter in part 5.
- `indep`（Optional）：DataFrame format, it has the same function as the parameter in part 5.
- `standard`（Optional）：Dict format, it has the same function as the parameter in part 5.
- `threshold`（Optional）：Float format, it has the same function as the parameter in part 6.
- `models`（Optional）：List format, Said to be involved in the analysis of the model, the default value is ['SVM','KNN','LR','RF','DT','XGB','STK'].
- `c`（Optional）：Float format, it has the same function as the parameter in part 6.
- `g`（Optional）：Float format, it has the same function as the parameter in part 6.
- `cv`（Optional）：Integer format, it has the same function as the parameter in part 8.
- `is_trend`（Optional）：Bool format, it has the same function as the parameter in part 6.
### Example
```python
classfier.evaluate(Train, Test=Test, indep=Indpt, models=['KNN','LR','RF','DT'])
classfier.evaluate_results
result = classfier.views(results=classfier.evaluate_results, title=' (CNV)', res=1)
result
```
![mark](http://img.frankgene.top/blog/20230413/L5J5r4mnHPdo.png)

![mark](http://img.frankgene.top/blog/20230413/V2cMdOxgRgYj.png)
## 14. Feature Selection with Different Models
```python
classfier.analyze(value, models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, standard=None, is_trend=True, rule='ANOVA', res='f1')
```
- `value`（Required）：DataFrame format, it has the same function as the parameter in part 5.
- `models`（Optional）：List format, it has the same function as the parameter in part 13.
- `c`（Optional）：Float format, it has the same function as the parameter in part 6.
- `g`（Optional）：Float format, it has the same function as the parameter in part 6.
- `standard`（Optional）：Dict format, it has the same function as the parameter in part 5.
- `is_trend`（Optional）：Bool format, it has the same function as the parameter in part 6.
- `rule`（Optional）：String format, it has the same function as the parameter in part 9.
- `res`（Optional）：String format, it has the same function as the parameter in part 9.
### Example
```python
classfier.analyze(Train, models=['KNN','LR','RF','DT'])
selection = classfier.select_results
selection
select_reslut = classfier.feature(results=selection)
select_reslut
```
![mark](http://img.frankgene.top/blog/20230413/XTvhC4CqEA8z.png)

![mark](http://img.frankgene.top/blog/20230413/rP3hr1M4fHed.png)
## 15. ROC Curves with Different Models
```python
classfier.multiroc(Train, Test, evalres=False, rocres=False, title='ROC_curve_all', out=os.getcwd(), models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, standard=None, is_trend=True)
```
- `Train`（Required）：DataFrame format, represents the input training set.
- `Test`（Required）：DataFrame format, represents the testing training set.
- `evalres`（Optional）：Dict format, indicates that Dtap uses the part13 function to parse the result. The default is False. If the value is passed in, this analysis is used in preference to drawing the picture, rather than recalculating with Train and Test.
- `rocres`（Optional）：Dict format, indicates that Dtap uses the part15 function to parse the result. The default is False. If the value is passed in, this analysis is used in preference to drawing the picture, rather than recalculating with Train and Test and evalres.
- `title`（Optional）：String format, it has the same function as the parameter in part 11.
- `out`（Optional）：String format, it has the same function as the parameter in part 11.
- `models`（Optional）：List format, it has the same function as the parameter in part 13.
- `c`（Optional）：Float format, it has the same function as the parameter in part 6.
- `g`（Optional）：Float format, it has the same function as the parameter in part 6.
- `standard`（Optional）：Dict format, it has the same function as the parameter in part 6.
- `is_trend`（Optional）：Bool format, it has the same function as the parameter in part 6.
9.
### Example 1
```python
classfier.multiroc(Train, Test, models=['KNN','LR','RF','DT'])
roc_result = classfier.roc_result
roc_result
```
![mark](http://img.frankgene.top/blog/20230413/1LDDUhhtKmeL.png)

![mark](http://img.frankgene.top/blog/20230413/YWGSspG9E7x9.png)
### Example 2
```python
roc_1 = classfier.multiroc(Train, Test, models=['KNN','LR','RF']).roc_result
roc_2 = classfier.multiroc(Train, Test, models=['KNN','LR','RF'], is_trend=False).roc_result
classfier.multiroc([], [], rocres={**roc_1, **roc_2})
```
![mark](http://img.frankgene.top/blog/20230413/fVWkRh6YXkLt.png)
## 16. PRC Curves with Different Models
```python
classfier.multiprc(Train, Test, evalres=False, prcres=False, title='PRC_curve_all', out=os.getcwd(), models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, standard=None, is_trend=True)
```
- `Train`（Required）：DataFrame format, represents the input training set.
- `Test`（Required）：DataFrame format, represents the testing training set.
- `evalres`（Optional）：Dict format, indicates that Dtap uses the part 13 function to parse the result. The default is False. If the value is passed in, this analysis is used in preference to drawing the picture, rather than recalculating with Train and Test.
- `prcres`（Optional）：Dict format, indicates that Dtap uses the part 16 function to parse the result. The default is False. If the value is passed in, this analysis is used in preference to drawing the picture, rather than recalculating with Train and Test and evalres.
- `title`（Optional）：String format, it has the same function as the parameter in part 11.
- `out`（Optional）：String format, it has the same function as the parameter in part 11.
- `models`（Optional）：List format, it has the same function as the parameter in part 13.
- `c`（Optional）：Float format, it has the same function as the parameter in part 6.
- `g`（Optional）：Float format, it has the same function as the parameter in part 6.
- `standard`（Optional）：Dict format, it has the same function as the parameter in part 6.
- `is_trend`（Optional）：Bool format, it has the same function as the parameter in part 6.
### Example 1
```python
classfier.multiprc(Train, Test, models=['KNN','LR','RF','DT'])
prc_result = classfier.prc_result
prc_result
```
![mark](http://img.frankgene.top/blog/20230413/2T6d6rhPCbcF.png)

![mark](http://img.frankgene.top/blog/20230413/sEdE5T5yMG1S.png)
### Example 2
```python
prc_1 = classfier.multiprc(Train, Test, models=['KNN','LR','RF']).prc_result
prc_2 = classfier.multiprc(Train, Test, models=['KNN','LR','RF'], is_trend=False).prc_result
classfier.multiprc([], [], prcres={**prc_1, **prc_2})
```
![mark](http://img.frankgene.top/blog/20230413/GcnJB8l9CCyX.png)
## 17. Weblogo for input data sets
```python
classfier.visual(out_path=os.getcwd(), res=0)
```
- `out_path`（Optional）：String format, represents the output path.
- `res`（Optional）：Integer format, represents the target data set. 0 is the training set, and 1 is the testing set.

# Dtap
disease trend analysis platform accurately predicts the occurrence of diseases under mixed background

## `block()`

对数据进行分块处理。

## `area()`

计算数据的面积。

## `trend(self, pos_value, neg_value)`

该函数的功能是分析正向和负向数据的趋势，将数据按照血液指标的正常范围分块，并计算每个区间的频数和面积，根据阈值筛选出有趋势的区间，并将结果存储在`self.pos_trend`和`self.neg_trend`中。

## `transform()`

对数据进行变换。

## `transform_predict()`

对变换后的数据进行预测。

## `information_value()`

计算数据的信息价值。

## `information()`

提取数据的信息特征。

## `information_predict()`

对信息特征进行预测。

## `onehot()`

对数据进行独热编码。

## `visual()`

对数据进行可视化展示。

## `svm_grid()`

使用支持向量机网格搜索最优参数。

## `V_to_L()`

将向量转换为列表。

## `calculate()`

计算数据的统计量。

## `fit()`

拟合模型。

## `crossv()`

使用交叉验证评估模型。

## `predict()`

对新数据进行预测。

## `sorted()`

对数据进行排序。

## `select()`

对数据进行筛选。

## `shap()`

使用SHAP值分析特征重要性。

## `roc()`

绘制ROC曲线和计算AUC值。

## `prc()`

绘制PRC曲线和计算AP值。

## `evaluate()`

评估模型的准确率、精确率、召回率、F1值等指标。

## `views()`

查看数据的基本信息和描述性统计量。

## `thresholdSearch()`

搜索最佳的分类阈值。

## `analyze()`

分析数据的相关性、分布、异常值等特征。

## `feature()`

提取或生成新的特征变量。

## `multiroc()`

绘制多分类问题的ROC曲线和计算AUC值。

## `multiprc()`

绘制多分类问题的PRC曲线和计算AP值。

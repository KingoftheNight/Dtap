# Dtap: 基于趋势向量的血常规疾病预测模型
# 作者: 梁雨朝
# 日期: 2022.07.10

# 导入包
import os
import math
import numpy as np
import pandas as pd
try:
    from .weblogo import weblogo
except:
    from weblogo import weblogo
import sklearn
import shap
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn import feature_selection
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from skrebate import TuRF
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
shap.initjs()

# 模型
class Dtap():
    def __init__(self):
        self.pos_trend = []
        self.neg_trend = []
        self.threshold = 0.2
        self.str_train = {'sequence': [], 'label': []}
        self.str_predict = {'sequence': [], 'label': []}
        self.index = []
        self.cv_result = {}
        self.model = KNeighborsClassifier()
        self.disType = ['N', 'P', 'D', 'H', 'C']
        self.infor_value_pos = []
        self.infor_value_neg = []
        self.standard = {
            'WBC': [3.5, 9.5],
            'NEUT#': [1.8, 6.3],
            'LYMPH#': [1.1, 3.2],
            'EO#': [0.02, 0.52],
            'BASO#': [0.0, 0.06],
            'MONO#': [0.1, 0.6],
            'NEUT%': [40.0, 75.0],
            'LYMPH%': [20.0, 50.0],
            'EO%': [0.4, 8.0],
            'BASO%': [0.0, 1.0],
            'MONO%': [3.0, 10.0],
            'RBC': [3.8, 5.8],
            'Hb': [115.0, 175.0],
            'HCT': [0.35, 0.5],
            'MCV': [82.0, 100.0],
            'MCH': [27.0, 34.0],
            'MCHC': [316.0, 350.0],
            'PLT': [125.0, 350.0],
            'PDW': [10.0, 18.0],
            'HGB': [110.0, 160.0],
            'MPV': [7.0, 13.0],
            'PCT': [0.1, 0.35],
            'R-CV': [10.9, 15.4]
        }
    
    def __repr__(self):
        return ""

    def block(self, standard_fs):
        min_fs = min(standard_fs)
        max_fs = max(standard_fs)
        gap = round((max_fs - min_fs) / 20, 2)
        content = []
        each_fs = []
        for i in range(20):
            content.append([round(min_fs + i * gap, 2), round(min_fs + (i + 1) * gap, 2)])
            each_fs.append(0)
        content[-1][-1] = max_fs + 0.01
        return content, each_fs

    def area(self, content, each_fs, k):
        out = []
        t = 0
        for i in range(len(content)):
            t += each_fs[i]
            if t >= sum(each_fs) * k:
                out.append(round(content[i][1], 2))
                break
        t = 0
        for i in range(len(content)):
            t += each_fs[i]
            if t >= sum(each_fs) * (1 - k):
                out.append(round(content[i][1], 2))
                break
        return out

    def trend(self, pos_value, neg_value):
        self.pos_trend = []
        for j in range(len(pos_value[0, :])):
            standard_fs = self.standard[self.index[j]]
            content, each_fs = self.block(standard_fs)
            for i in range(len(pos_value)):
                for n in range(len(each_fs)):
                    if content[n][0] <= pos_value[i, j] < content[n][1]:
                        each_fs[n] += 1
            self.pos_trend.append([standard_fs[0]] + self.area(content, each_fs, self.threshold) + [standard_fs[-1]])
        self.neg_trend = []
        for j in range(len(neg_value[0, :])):
            standard_fs = self.standard[self.index[j]]
            content, each_fs = self.block(standard_fs)
            for i in range(len(neg_value)):
                for n in range(len(each_fs)):
                    if content[n][0] <= neg_value[i, j] < content[n][1]:
                        each_fs[n] += 1
            self.neg_trend.append([standard_fs[0]] + self.area(content, each_fs, self.threshold) + [standard_fs[-1]])

    def transfrom(self, value):
        self.str_train = {'sequence': [], 'label': []}
        for j in range(len(value)):
            line = list(value.iloc[j, :])
            mid = ''
            for i in range(len(line[:-1])):
                if self.pos_trend[i][1] <= line[i] <= self.pos_trend[i][2]:
                    mid += 'C'
                elif self.pos_trend[i][0] <= line[i] < self.pos_trend[i][1]:
                    mid += 'D'
                elif self.pos_trend[i][2] < line[i] <= self.pos_trend[i][3]:
                    mid += 'H'
                elif line[i] > self.pos_trend[i][3]:
                    mid += 'P'
                elif line[i] < self.pos_trend[i][0]:
                    mid += 'N'
            self.str_train['sequence'].append(mid)
            self.str_train['label'].append(line[-1])

    def transfrom_predict(self, value):
        self.str_predict = {'sequence': [], 'label': []}
        for j in range(len(value)):
            line = list(value.iloc[j, :])
            mid = ''
            for i in range(len(line[:-1])):
                if self.pos_trend[i][1] <= line[i] <= self.pos_trend[i][2]:
                    mid += 'C'
                elif self.pos_trend[i][0] <= line[i] < self.pos_trend[i][1]:
                    mid += 'D'
                elif self.pos_trend[i][2] < line[i] <= self.pos_trend[i][3]:
                    mid += 'H'
                elif line[i] > self.pos_trend[i][3]:
                    mid += 'P'
                elif line[i] < self.pos_trend[i][0]:
                    mid += 'N'
            self.str_predict['sequence'].append(mid)
            self.str_predict['label'].append(line[-1])

    def information_value(self, value):
        H = 0
        for i in value:
            if i != 0:
                H += i * math.log(i, 2)
        Rseq = math.log(len(value), 2) + H
        out = []
        for i in value:
            if i != 0:
                out.append(Rseq * i)
            else:
                out.append(0)
        return out

    def information(self, data):
        # 频数
        count_pos, count_neg = [], []
        for i in range(len(data['sequence'][0])):
            mid_pos, mid_neg = list(0 for j in range(len(self.disType))), list(0 for j in range(len(self.disType)))
            for j in range(len(data['sequence'])):
                if data['sequence'][j][i] in self.disType:
                    if data['label'][j] == 1:
                        mid_pos[self.disType.index(data['sequence'][j][i])] += 1
                    else:
                        mid_neg[self.disType.index(data['sequence'][j][i])] += 1
            count_pos.append(mid_pos)
            count_neg.append(mid_neg)
        count_pos = np.array(count_pos)
        count_neg = np.array(count_neg)
        # 频率
        self.infor_value_pos, self.infor_value_neg = [], []
        for i in range(len(count_pos)):
            mid = []
            for j in range(len(count_pos[i])):
                mid.append(count_pos[i, j] / np.sum(count_pos[i]))
            # 转信息熵
            self.infor_value_pos.append(self.information_value(mid))
        for i in range(len(count_neg)):
            mid = []
            for j in range(len(count_neg[i])):
                mid.append(count_neg[i, j] / np.sum(count_neg[i]))
            # 转信息熵
            self.infor_value_neg.append(self.information_value(mid))
        # 转换
        infor_data = []
        for i in range(len(data['sequence'])):
            mid = list(0 for j in range(len(data['sequence'][i])))
            if data['label'][i] == 1:
                for j in range(len(data['sequence'][i])):
                    mid[j] = self.infor_value_pos[j][self.disType.index(data['sequence'][i][j])]
            else:
                for j in range(len(data['sequence'][i])):
                    mid[j] = self.infor_value_neg[j][self.disType.index(data['sequence'][i][j])]
            infor_data.append(mid)
        self.str_train['information'] = infor_data

    def information_predict(self, data):
        infor_data = []
        for i in range(len(data['sequence'])):
            mid = list(0 for j in range(len(data['sequence'][i])))
            if data['label'][i] == 1:
                for j in range(len(data['sequence'][i])):
                    mid[j] = self.infor_value_pos[j][self.disType.index(data['sequence'][i][j])]
            else:
                for j in range(len(data['sequence'][i])):
                    mid[j] = self.infor_value_neg[j][self.disType.index(data['sequence'][i][j])]
            infor_data.append(mid)
        self.str_predict['information'] = infor_data
    
    def onehot(self, data):
        # 转换
        infor_data = []
        for i in range(len(data['sequence'])):
            mid = []
            for j in range(len(data['sequence'][i])):
                if data['sequence'][i][j] == 'N':
                    mid.append(0.2)
                elif data['sequence'][i][j] == 'D':
                    mid.append(0.4)
                elif data['sequence'][i][j] == 'C':
                    mid.append(0.6)
                elif data['sequence'][i][j] == 'H':
                    mid.append(0.8)
                elif data['sequence'][i][j] == 'P':
                    mid.append(1.0)
            infor_data.append(mid)
        return infor_data

    def visual(self, out_path=os.getcwd(), seq=False, res=0):
        if res == 0:
            if seq:
                weblogo(data=seq, tp='d', x_label=self.index, out=out_path)
            else:
                pos_seq, neg_seq = [], []
                for i in range(len(self.str_train['sequence'])):
                    if self.str_train['label'][i] == 1:
                        pos_seq.append(self.str_train['sequence'][i])
                    else:
                        neg_seq.append(self.str_train['sequence'][i])
                weblogo(data=pos_seq, tp='d', x_label=self.index, out=out_path+'-positive.svg')
                weblogo(data=neg_seq, tp='d', x_label=self.index, out=out_path+'-negative.svg')
        elif res == 1:            
            if seq:
                weblogo(data=seq, tp='d', x_label=self.index, out=out_path)
            else:
                pos_seq, neg_seq = [], []
                for i in range(len(self.str_predict['sequence'])):
                    if self.str_predict['label'][i] == 1:
                        pos_seq.append(self.str_predict['sequence'][i])
                    else:
                        neg_seq.append(self.str_predict['sequence'][i])
                weblogo(data=pos_seq, tp='d', x_label=self.index, out=out_path+'-positive.svg')
                weblogo(data=neg_seq, tp='d', x_label=self.index, out=out_path+'-negative.svg')

    def svm_grid(self, train_data, train_label):
        my_svm = svm.SVC(decision_function_shape="ovo", random_state=0)
        c_number = []
        for i in range(-5, 15 + 1, 2):
            c_number.append(2 ** i)
        gamma = []
        for i in range(-15, 3 + 1, 2):
            gamma.append(2 ** i)
        parameters = {'C': c_number, 'gamma': gamma}
        new_svm = GridSearchCV(my_svm, parameters, cv=5, scoring="accuracy", return_train_score=False, n_jobs=1)
        model = new_svm.fit(train_data, train_label)
        best_c = model.best_params_['C']
        best_g = model.best_params_['gamma']
        return best_c, best_g

    def V_to_L(self, value, threshold=0.5):
        out = []
        for i in value:
            if i[0] >= threshold:
                out.append(0)
            else:
                out.append(1)
        return np.array(out)
    
    def calculate(self, y_train, y_predict):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(y_train)):
            t_label, p_label = y_train[i], y_predict[i]
            if t_label == p_label == 1:
                tp += 1
            if t_label == p_label == 0:
                tn += 1
            if t_label == 1 and p_label == 0:
                fn += 1
            if t_label == 0 and p_label == 1:
                fp += 1
        try:
            acc = (tp + tn) / (tp + tn + fp + fn)
        except:
            acc = 0
        try:
            sn = tp / (tp + fn)
        except:
            sn = 0
        try:
            sp = tn / (tn + fp)
        except:
            sp = 0
        try:
            ppv = tp / (tp + fp)
        except:
            ppv = 0
        try:
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp))
        except:
            mcc = 0
        try:
            f1 = (2 * ppv * sn) / (ppv + sn)
        except:
            f1 = 0
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'acc': acc, 'sn': sn, 'sp': sp, 'ppv': ppv, 'mcc': mcc, 'f1': f1}

    def fit(self, value, standard=None, threshold=None, model='SVM', c=None, g=None, is_trend=True):
        # 更新参数
        if standard != None:
            self.standard = standard
        if threshold != None:
            self.threshold = threshold
        # 数据预处理
        self.index = list(value.columns)[:-1]
        if is_trend:
            pos_value = np.array(value[value['label'].isin([1])].iloc[:, :-1])
            neg_value = np.array(value[value['label'].isin([0])].iloc[:, :-1])
            # 计算趋势
            self.trend(pos_value, neg_value)
            # 提取趋势向量
            self.transfrom(value)
            self.str_train['information'] = self.onehot(self.str_train)
        else:
            self.str_train = {'information': [], 'label': []}
            self.str_train['label'] = list(list(value.iloc[j, :])[-1] for j in range(len(value)))
            self.str_train['information'] = list(list(item) for item in MinMaxScaler().fit_transform(value.iloc[:, :-1]))
        # 构建模型
        if model == 'SVM':
            if c is None and g is None:
                best_c, best_g = self.svm_grid(self.str_train['information'], self.str_train['label'])
            else:
                best_c, best_g = c, g
            self.model = svm.SVC(kernel='rbf', C=best_c, gamma=best_g, probability=True)
            self.model.fit(self.str_train['information'], self.str_train['label'])
        elif model == 'KNN':
            self.model = KNeighborsClassifier()
            self.model.fit(self.str_train['information'], self.str_train['label'])
        elif model == 'LR':
            self.model = sklearn.linear_model.LogisticRegression()
            self.model.fit(self.str_train['information'], self.str_train['label'])
        elif model == 'DT':
            self.model = sklearn.tree.DecisionTreeClassifier(min_samples_split=2)
            self.model.fit(self.str_train['information'], self.str_train['label'])
        elif model == 'RF':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=None,
                                                                 min_samples_split=2, random_state=0)
            self.model.fit(self.str_train['information'], self.str_train['label'])
        elif model == 'XGB':
            self.model = XGBClassifier(eta=0.01, objective='binary:logistic', subsample=0.5, use_label_encoder=False,
                                       base_score=np.mean(self.str_train['label']), eval_metric='logloss')
            self.model.fit(self.str_train['information'], self.str_train['label'])
        elif model == 'STK':
            if c is None and g is None:
                best_c, best_g = self.svm_grid(self.str_train['information'], self.str_train['label'])
            else:
                best_c, best_g = c, g
            self.model = {
                    'SVM': svm.SVC(kernel='rbf', C=best_c, gamma=best_g, probability=True),
                    'KNN': KNeighborsClassifier(),
                    'LR': sklearn.linear_model.LogisticRegression(),
                    'DT': sklearn.tree.DecisionTreeClassifier(min_samples_split=2),
                    'RF': RandomForestClassifier(n_estimators=100, max_depth=None,
                                                 min_samples_split=2, random_state=0),
                    'XGB': XGBClassifier(eta=0.01, objective='binary:logistic', subsample=0.5, use_label_encoder=False, base_score=np.mean(self.str_train['label']), eval_metric='logloss')
                }
            for key in self.model:
                self.model[key].fit(self.str_train['information'], self.str_train['label'])
        return self

    def crossv(self, cv=5, is_select=False):
        if is_select:
            if type(self.model) == dict:
                y_predict = list(0 for i in range(len(list(self.str_select['label']))))
                for key in self.model:
                    y_predict = [x + y for x,y in zip(y_predict, cross_val_predict(self.model[key], self.str_select['information'], self.str_select['label'], cv=cv))]
                y_predict = list(round(x/len(self.model)) for x in y_predict)
            else:
                y_predict = cross_val_predict(self.model, self.str_select['information'], self.str_select['label'], cv=cv)
            self.cv_result = self.calculate(list(self.str_select['label']), list(y_predict))
        else:
            if type(self.model) == dict:
                y_predict = list(0 for i in range(len(list(self.str_train['label']))))
                for key in self.model:
                    y_predict = [x + y for x,y in zip(y_predict, cross_val_predict(self.model[key], self.str_train['information'], self.str_train['label'], cv=cv))]
                y_predict = list(round(x/len(self.model)) for x in y_predict)
            else:
                y_predict = cross_val_predict(self.model, self.str_train['information'], self.str_train['label'], cv=cv)
            self.cv_result = self.calculate(list(self.str_train['label']), list(y_predict))
        return self

    def predict(self, value, threshold=0.5, is_trend=True, weight=0.024, model=None):
        if 'label' not in value:
            value['label'] = list(0 for i in range(len(value)))
        if model != None:
            self.model = model
        # 数据预处理
        if list(value.columns)[:-1] != self.index:
            print('Disease labels dismatch!')
            return
        if is_trend:
            # 提取趋势向量
            self.transfrom_predict(value)
            self.str_predict['information'] = self.onehot(self.str_predict)
        else:
            self.str_predict = {'information': [], 'label': []}
            self.str_predict['label'] = list(list(value.iloc[j, :])[-1] for j in range(len(value)))
            self.str_predict['information'] = list(list(item) for item in MinMaxScaler().fit_transform(value.iloc[:, :-1]))
        # 预测
        if type(self.model) == dict:
            y_predict = list(0 for i in range(len(list(self.str_predict['label']))))
            for key in self.model:
                y_predict = [x + y[0]*(round(y[0], 1)+weight) for x,y in zip(y_predict, self.model[key].predict_proba(self.str_predict['information']))]
            self.predict_value = np.array(list([round(x/len(self.model), 3), 1-round(x/len(self.model), 3)] for x in y_predict))
        else:
            self.predict_value = self.model.predict_proba(self.str_predict['information'])
        y_predict = self.V_to_L(self.predict_value, threshold)
        self.predict_label = y_predict
        self.predict_result = self.calculate(list(self.str_predict['label']), list(y_predict))
        return self

    def sorted(self, data):
        arr = []
        for i in data:
            arr.append(i)
        index = []
        for i in range(len(arr)):
            index.append(i)
        for i in range(len(arr) - 1):
            min_index = i
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[min_index]:
                    min_index = j
            index[min_index], index[i] = index[i], index[min_index]
            arr[min_index], arr[i] = arr[i], arr[min_index]
        # 倒序输出
        re_index = []
        for i in range(len(index) - 1, -1, -1):
            re_index.append(index[i])
        return re_index
    
    def select(self, rule='ANOVA', res='f1'):
        # 特征排序
        if rule == 'ANOVA':
            self.selection = feature_selection.SelectPercentile(feature_selection.f_classif, percentile=100)
            self.selection.fit_transform(self.str_train['information'], self.str_train['label'])
            self.selection.value = list(self.selection.scores_)
            self.selection.order = self.sorted(list(self.selection.scores_))
        elif rule == 'Correlation':
            self.selection = feature_selection.SelectPercentile(feature_selection.r_regression, percentile=100)
            self.selection.fit_transform(self.str_train['information'], self.str_train['label'])
            self.selection.value = list(self.selection.scores_)
            self.selection.order = self.sorted(list(self.selection.scores_))
        elif rule == 'TURF':
            fs = TuRF(core_algorithm="ReliefF", n_features_to_select=2, pct=0.5,verbose=True).fit(np.array(self.str_train['information']), np.array(self.str_train['label']), self.index)
            self.selection.value = list(fs.feature_importances_)
            self.selection.order = self.sorted(list(self.selection.value))
        elif rule == 'PCA-SVD':
            pca = PCA(n_components=1)
            pca.fit(np.array(self.str_train['information']))
            pc1_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            pc1_featurescore = pd.DataFrame({'Feature': self.index, 'PC1_loading': pc1_loadings.T[0], 'PC1_loading_abs': abs(pc1_loadings.T[0])})
            self.selection.value = list(i for i in pc1_featurescore['PC1_loading_abs'])
            self.selection.order = self.sorted(list(self.selection.value))
        elif rule == 'RF-Weight':
            forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
            forest.fit(np.array(self.str_train['information']), np.array(self.str_train['label']))
            self.selection.value = list(forest.feature_importances_)
            self.selection.order = self.sorted(list(self.selection.value))
        elif rule == 'Lasso':
            lasso = Lasso(alpha=10**(-3))
            model_lasso = lasso.fit(np.array(self.str_train['information']), np.array(self.str_train['label']))
            coef = pd.Series(model_lasso.coef_,index=self.index)
            self.selection.value = list(coef)
            self.selection.order = self.sorted(list(self.selection.value))
        # 增量特征交叉验证
        self.selection.performance, self.selection.max_fs, self.selection.max_fv = [], 0, 0
        self.str_select = {'label': self.str_train['label'], 'information': list([] for i in range(len(self.str_train['label'])))}
        tt = 0
        for i in self.selection.order:
            tt += 1
            for j in range(len(self.str_train['label'])):
                self.str_select['information'][j].append(self.str_train['information'][j][i])
            # 模型性能评估
            self.crossv(cv=5, is_select=True)
            self.selection.performance.append(self.cv_result)
            if self.cv_result[res] >= self.selection.max_fv:
                self.selection.max_fs, self.selection.max_fv = tt, self.cv_result[res]
            print('\r' + str(tt) + "~" + str(len(self.selection.order)), end='', flush=True)
        # 最优特征集
        self.selection.best_feature = {'label': self.str_train['label'], 'index': [], 'information': list([] for i in range(len(self.str_train['label'])))}
        for i in self.selection.order[:self.selection.max_fs]:
            self.selection.best_feature['index'].append(self.index[i])
            for j in range(len(self.str_train['label'])):
                self.selection.best_feature['information'][j].append(self.str_train['information'][j][i])
        return self
    
    def shap(self, item=0, plot=['bar', 'beeswarm', 'heatmap', 'force'], out=os.getcwd()):
        print('Starting analyze with SHAP')
        cmap = LinearSegmentedColormap.from_list('Dtap', ['#6496AA', '#FFFFFF', '#CA6F64'])
        shap_model = sklearn.linear_model.LogisticRegression()
        shap_model.fit(np.array(self.str_train['information']), np.array(self.str_train['label']))
        explainer = shap.Explainer(shap_model, np.array(self.str_train['information']), feature_names=self.index)
        self.shap_values = explainer(np.array(self.str_predict['information']))
        if 'bar' in plot:
            print('Plot shap-bar chart')
            shap.plots.bar(self.shap_values, show=False, max_display=60)
            plt.savefig(os.path.join(out, 'bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
        if 'beeswarm' in plot:
            print('Plot shap-beeswarm chart')
            shap.plots.beeswarm(self.shap_values, color=cmap, show=False, plot_size=[8,5], max_display=10)
            plt.savefig(os.path.join(out, 'beeswarm.png'), dpi=300, bbox_inches='tight')
            plt.close()
        if 'heatmap' in plot:
            print('Plot shap-heatmap chart')
            shap.plots.heatmap(self.shap_values, cmap=cmap, show=False, max_display=10)
            plt.savefig(os.path.join(out, 'heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        if 'force' in plot:
            print('Plot shap-force chart')
            shap.plots.force(self.shap_values[item], matplotlib=True, show=False)
            plt.savefig(os.path.join(out, 'force.png'), dpi=300, bbox_inches='tight')
            plt.close()
        print('Processing finished!')
        return self

    def roc(self, all_tpr=None, all_fpr=None, label='ROC', color=['#525288','#F17666','#624941','#B90000','#D7A100','#1A6840','#BFAEA4'], lw=1, figsize=[5,5], dpi=300, xlabel='False Positive Rate', ylabel='True Positive Rate', legend="lower right", title='ROC_curve', out=os.getcwd()):
        if all_tpr == None and all_fpr == None:
            self.fpr, self.tpr, thresholds = roc_curve(np.array(self.str_predict['label']), np.array(list(x[1] for x in self.predict_value)))
            if out == False:
                return self
            roc_auc = auc(self.fpr, self.tpr)
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(self.fpr, self.tpr, color=color[0], lw=lw, label=label+" (AUROC = %0.2f)" % roc_auc)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc=legend)
            plt.title(title)
            plt.savefig(os.path.join(out, title + '.png'))
        elif all_tpr != None and all_fpr != None:
            plt.figure(figsize=figsize, dpi=dpi)
            for i in range(len(all_tpr)):
                roc_auc = auc(all_fpr[i], all_tpr[i])
                plt.plot(all_fpr[i], all_tpr[i], color=color[i], lw=lw, label=label[i]+" (AUROC = %0.2f)" % roc_auc)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc=legend)
            plt.title(title)
            plt.savefig(os.path.join(out, title + '.png'))
        return self
    
    def prc(self, all_pre=None, all_rec=None, index=None, value=None, label='PRC', color=['#525288','#F17666','#624941','#B90000','#D7A100','#1A6840','#BFAEA4'], lw=1, figsize=[5,5], dpi=300, xlabel='Precision', ylabel='Recall', legend="lower left", title='PRC_curve', out=os.getcwd()):
        if all_pre == None and all_rec == None and index == None and value == None:
            self.pre, self.rec, thresholds = precision_recall_curve(np.array(self.str_predict['label']), np.array(list(x[1] for x in self.predict_value)))
            if out == False:
                return self
            prc_auprc = average_precision_score(np.array(self.str_predict['label']), np.array(list(x[1] for x in self.predict_value)))
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(self.pre, self.rec, color=color[0], lw=lw, label=label+" (AUPRC = %0.2f)" % prc_auprc)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc=legend)
            plt.title(title)
            plt.savefig(os.path.join(out, title + '.png'))
        elif all_pre != None and all_rec != None and index != None and value != None:
            plt.figure(figsize=figsize, dpi=dpi)
            for i in range(len(all_rec)):
                prc_auprc = average_precision_score(np.array(index[i]), np.array(list(x[1] for x in value[i])))
                plt.plot(all_pre[i], all_rec[i], color=color[i], lw=lw, label=label[i]+" (AUPRC = %0.2f)" % prc_auprc)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc=legend)
            plt.title(title)
            plt.savefig(os.path.join(out, title + '.png'))
        return self
    
    def evaluate(self, Train, Test=False, indep=False, standard=None, threshold=None, models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, cv=5, is_trend=True):
        self.evaluate_results = {}
        self.evaluate_orignal = {key: {'value':[], 'label':[]} for key in models}
        print('Starting evaluate models')
        for key in models:
            print(key)
            self.fit(Train, standard=standard, threshold=threshold, model=key, c=c, g=g, is_trend=is_trend)
            self.crossv(cv=5)
            self.evaluate_results[key] = [self.cv_result]
            if type(Test) != bool:
                self.predict(Test, is_trend=is_trend)
                self.evaluate_results[key].append(self.predict_result)
                self.evaluate_orignal[key]['value'] = [self.predict_value]
                self.evaluate_orignal[key]['label'] = self.str_predict['label']
            if type(indep) != bool:
                self.predict(indep, is_trend=is_trend)
                self.evaluate_results[key].append(self.predict_result)
                self.evaluate_orignal[key]['value'].append(self.predict_value)
        print('Processing finished!')
        return self
    
    def views(self, results=None, res=0, out='numpy', title=''):
        if results:
            self.evaluate_results = results
        def view(t, v):
            lab, out = [], []
            for key in self.evaluate_results:
                lab.append(key+title)
                out.append(round(self.evaluate_results[key][t][v], 3))
            return lab, out
        data = [['']+view(res, 'acc')[0]]
        data.append(['ACC']+view(res, 'acc')[1])
        data.append(['SP']+view(res, 'sp')[1])
        data.append(['SN']+view(res, 'sn')[1])
        data.append(['PPV']+view(res, 'ppv')[1])
        data.append(['MCC']+view(res, 'mcc')[1])
        data.append(['F1']+view(res, 'f1')[1])
        if out == 'numpy':
            return np.array(data).T
        elif out == 'csv':
            return '\n'.join(list(','.join(item) for item in np.array(data).T.tolist()))
    
    def thresholdSearch(self, Train, Test=False, indep=False, res=0, order='mcc', standard=None, model='RF'):
        k_resault = {}
        print('Starting search best k from 0.04 to 0.96')
        for i in range(1, 25):
            k_resault[i/25] = []
            self.fit(Train, standard=standard, threshold=i/25, model=model)
            self.crossv(cv=5)
            k_resault[i/25].append(self.cv_result)
            if type(Test) != bool:
                self.predict(Test)
                k_resault[i/25].append(self.predict_result)
            if type(indep) != bool:
                self.predict(indep)
                k_resault[i/25].append(self.predict_result)
            print('k='+str(i/25)+'\t'+order+'='+str(round(k_resault[i/25][res][order], 3)))
        # 保存最优结果
        best_k, best_performance = 0, 0
        for key in k_resault:
            if k_resault[key][res][order] >= best_performance:
                best_performance = k_resault[key][res][order]
                best_k = key
        self.threshold = best_k
        # 展示结果
        print('\nbest k='+str(best_k))
        return self

    def analyze(self, Train, models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, standard=None, is_trend=True, rule='ANOVA', res='f1'):
        self.select_results = {
                'performance': {key: [] for key in models},
                'max_fv': {key: 0 for key in models},
                'max_fs': {key: 0 for key in models},
                'best_feature': {key: {} for key in models},
                'order': [],
                'scores': [],
                'pvalue': [],
                'rule': rule,
                'res': res,
                'istrend': is_trend
            }
        print('Starting analyze feature importance with different models')
        for key in models:
            print('')
            print(key+' method='+rule)
            self.fit(Train, model=key, c=c, g=g, is_trend=is_trend, standard=standard)
            self.select(rule=rule, res=res)
            self.select_results['performance'][key] = self.selection.performance
            self.select_results['max_fv'][key] = self.selection.max_fv
            self.select_results['max_fs'][key] = self.selection.max_fs
            self.select_results['order'] = self.selection.order
            self.select_results['scores'] = list(self.selection.scores_)
            self.select_results['pvalue'] = list(self.selection.pvalues_)
            self.select_results['best_feature'][key] = self.selection.best_feature
        print('')
        print('\nProcessing finished!')
        return self
    
    def feature(self, results=None, out='numpy'):
        if results:
            self.select_results = results
        # 展示结果
        performance = [['']+list(self.index[i] for i in self.select_results['order'])]
        for key in self.select_results['performance']:
            performance.append([key] + list(line[self.select_results['res']] for line in self.select_results['performance'][key]))
        performance = np.array(performance).T
        
        weight = [['Feature']+list(key for key in self.index)]
        weight.append([self.select_results['rule']]+self.select_results['scores'])
        weight = np.array(weight).T
        
        total = [['Models', 'Feature', self.select_results['res']]]
        for key in self.select_results['performance']:
            total.append([key, self.select_results['max_fs'][key], round(self.select_results['max_fv'][key], 3)])  
        total = np.array(total)
        if out == 'numpy':
            return performance, weight, total
        elif out == 'csv':
            return '\n'.join(list(','.join(item) for item in performance.tolist())), '\n'.join(list(','.join(item) for item in weight.tolist())), '\n'.join(list(','.join(item) for item in total.tolist()))
        
    def multiroc(self, Train, Test, rocres=False, title='ROC_curve_all', out=os.getcwd(), models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, standard=None, is_trend=True):
        print('Starting plot ROC curves with different models')
        if rocres:
            self.roc_result = rocres
        else:
            if is_trend:
                self.roc_result = {key + ' (trend)': [] for key in models}
            else:
                self.roc_result = {key + ' (base)': [] for key in models}
            for key in self.roc_result:
                print(key)
                self.fit(Train, model=key.split(' ')[0], c=c, g=g, standard=standard, is_trend=is_trend)
                self.predict(Test)
                self.roc(label='multy', out=False)
                self.roc_result[key].append(self.fpr)
                self.roc_result[key].append(self.tpr)
        self.all_fpr=list(self.roc_result[key][0] for key in self.roc_result)
        self.all_tpr=list(self.roc_result[key][1] for key in self.roc_result)
        self.roc(all_fpr=self.all_fpr, all_tpr=self.all_tpr, label=list(key for key in self.roc_result), title=title, out=out)
        print('Processing finished!')
        return self
    
    def multiprc(self, Train, Test, prcres=False, title='PRC_curve_all', out=os.getcwd(), models=['SVM','KNN','LR','RF','DT','XGB','STK'], c=None, g=None, standard=None, is_trend=True):
        print('Starting plot PRC curves with different models')
        if prcres:
            self.prc_result = prcres
        else:
            if is_trend:
                self.prc_result = {key + ' (trend)': [] for key in models}
            else:
                self.prc_result = {key + ' (base)': [] for key in models}
            for key in self.prc_result:
                print(key)
                self.fit(Train, model=key.split(' ')[0], c=c, g=g, standard=standard, is_trend=is_trend)
                self.predict(Test)
                self.prc(label='multy', out=False)
                self.prc_result[key].append(self.pre)
                self.prc_result[key].append(self.rec)
                self.prc_result[key].append(self.str_predict['label'])
                self.prc_result[key].append(self.predict_value)
        self.all_pre=list(self.prc_result[key][0] for key in self.prc_result)
        self.all_rec=list(self.prc_result[key][1] for key in self.prc_result)
        self.prc(all_pre=self.all_pre, all_rec=self.all_rec, index=list(self.prc_result[key][2] for key in self.prc_result), value=list(self.prc_result[key][3] for key in self.prc_result), label=list(key for key in self.prc_result), title=title, out=out)
        print('Processing finished!')
        return self

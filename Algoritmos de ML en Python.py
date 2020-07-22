#!/usr/bin/env python
# coding: utf-8

# #  <center> <font color='darkcyan'>  Diferentes Técnicas de Aprendizaje Estadístico </font> </center>
# 
# ###   <center> <font color='darkcyan'> Jessica Quintero López </font> </center>

# ###  <font color='darkcyan'>  Importamos los paquetes necesarios: </font> 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn import svm
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets


# #  <center> <font color='darkcyan'> Regresión Lineal </font> </center>

# In[10]:


base1 = pd.read_csv('USA_Housing.csv')
base1.head()


# In[11]:


base1.columns


# In[3]:


print("Tipo de Variables \n")
print(base1.dtypes)
print("Cantidad de datos faltantes \n")
print(base1.isnull().sum())
print("Estadisticos de resumen \n")
print(base1.describe(include="all"))


# In[12]:


x = base1.drop('Address', axis=1)
y = base1['Price']
x2 = x.drop('Avg. Area Number of Bedrooms', axis=1)


# In[14]:


sns.pairplot(x2, corner=True, diag_kind="kde")


# In[76]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=45)


# In[77]:


rl = LinearRegression()
rl.fit(x_train,y_train)


# In[78]:


print("Coeficientes \n")
rl.coef_


# In[79]:


predicciones_rl = rl.predict(x_test)
predicciones_rl[0:20]


# In[80]:


print("Error cuadrático medio \n")
metrics.mean_squared_error(y_test, predicciones_rl)


# In[81]:


print("RMSE \n")
np.sqrt(metrics.mean_squared_error(y_test, predicciones_rl))


# In[33]:


sns.heatmap(x.corr(), annot=True)


# #  <center> <font color='darkcyan'> Regresión Logistica </font> </center>

# In[86]:


base2 = pd.read_csv('breast_cancer_dataset.csv')
base2.head()


# In[87]:


print("Tipo de Variables \n")
print(base2.dtypes)
print("Cantidad de datos faltantes \n")
print(base2.isnull().sum())
print("Estadisticos de resumen \n")
print(base2.describe(include="all"))


# In[88]:


## convirtiendo la variable class en categorica
base2['class'] = base2['class'].astype('category')
x = base2.drop('class', axis=1)
y = base2['class']


# In[89]:


sns.heatmap(base2.corr(), annot=True)


# In[90]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=45)
rlog = LogisticRegression()
rlog.fit(x_train, y_train)


# In[91]:


predicciones_log = rlog.predict(x_test)
predicciones_log[0:11]


# In[92]:


print("Accuracy \n")
metrics.accuracy_score(y_test, predicciones_log)


# In[93]:


print("ROC \n")
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicciones_log, pos_label=2)
metrics.auc(fpr, tpr)


# In[94]:


print("Matriz de confusión \n")
metrics.confusion_matrix(y_test, predicciones_log)


# In[95]:


print("Reporte General del Ajuste \n")
print(metrics.classification_report(y_test, predicciones_log))


# #  <center> <font color='darkcyan'> Árboles de decisión </font> </center>

# In[96]:


base2 = pd.read_csv('breast_cancer_dataset.csv')
base2.head()


# In[97]:


base2['class'] = base2['class'].astype('category')


# In[98]:


x = base2.drop('class', axis=1)
y = base2['class']


# In[99]:


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=45)
ad = DecisionTreeClassifier()
ad.fit(x_train,y_train)


# In[100]:


predi_ad = ad.predict(x_test)
predi_ad[0:11]


# In[103]:


print('Accuracy \n')
metrics.accuracy_score(y_test,predi_ad)


# In[104]:


print('Matriz de confusión \n')
metrics.confusion_matrix(y_test, predi_ad)


# In[105]:


print("Reporte General del Ajuste \n")
print(metrics.classification_report(y_test, predi_ad ))


# #  <center> <font color='darkcyan'> RandomForest </font> </center>

# In[106]:


base2 = pd.read_csv('breast_cancer_dataset.csv')
base2.head()


# In[107]:


base2['class'] = base2['class'].astype('category')
x = base2.drop('class', axis=1)
y = base2['class']


# In[108]:


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=45)
rf = RandomForestClassifier()
rf.fit(x_train,y_train)


# In[109]:


pred_rf = rf.predict(x_test)


# In[110]:


accuracy = []
for i in range(50,100):
    rf = RandomForestClassifier(n_estimators=i)
    rf.fit(x_train, y_train)
    prediccion_i = rf.predict(x_test)
    accuracy.append(metrics.balanced_accuracy_score(y_test, prediccion_i))


# In[111]:


accuracy


# In[112]:


rf = RandomForestClassifier(n_estimators=73)
rf.fit(x_train, y_train)
prediccion_rf = rf.predict(x_test)


# In[113]:


print('Matriz de confusión \n')
metrics.confusion_matrix(y_test, prediccion_rf)


# In[114]:


print('Accuracy \n')
metrics.accuracy_score(y_test, prediccion_rf)


# In[116]:


print("Reporte General del Ajuste \n")
print(metrics.classification_report(y_test, prediccion_rf))


# #  <center> <font color='darkcyan'> KNN </font> </center>

# In[59]:


base2 = pd.read_csv('breast_cancer_dataset.csv')
base2.head()


# In[60]:


base2['class'] = base2['class'].astype('category')
x = base2.drop('class', axis=1)
y = base2['class']
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=45)
kn =  KNeighborsClassifier()
kn.fit(x_train,y_train)


# In[61]:


kn =  KNeighborsClassifier(n_neighbors=14)
kn.fit(x_train,y_train)
predi_kn = kn.predict(x_test)


# In[62]:


print("accuracy \n")
metrics.accuracy_score(y_test, predi_kn)


# In[117]:


print("Reporte General del Ajuste \n")
print(metrics.classification_report(y_test, predi_kn))


# #  <center> <font color='darkcyan'> Análisis Discriminante Lineal y Análisis Cuadratico Lineal  </font> </center>

# In[63]:


base2 = pd.read_csv('breast_cancer_dataset.csv')
base2.head()


# In[64]:


base2['class'] = base2['class'].astype('category')
x = base2.drop('class', axis=1)
y = base2['class']
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=45)


# In[65]:


lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
y_pred = lda.predict(x_test)
print("Matriz de confusión \n")
print(metrics.confusion_matrix(y_test, y_pred))
print("Accuracy \n")
print(metrics.accuracy_score(y_test, y_pred))


# In[118]:


print("Reporte General del Ajuste LDA \n")
print(metrics.classification_report(y_test, y_pred))


# In[66]:


qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train,y_train)
y_pred = qda.predict(x_test)
print("Matriz de confusión \n")
print(metrics.confusion_matrix(y_test, y_pred))
print("Accuracy \n")
print(metrics.accuracy_score(y_test, y_pred))


# In[119]:


print("Reporte General del Ajuste QDA \n")
print(metrics.classification_report(y_test, y_pred))


# #  <center> <font color='darkcyan'> Máquinas de Soporte Vectorial  </font> </center>

# In[197]:


np.random.seed(20200327)
x1 = np.random.normal(1, 0.1, 100)
x2 = np.random.normal(1, 0.1, 100) 
x1[0:30] = x1[0:30] + 1
x1[31:60] = x1[31:60] - 1
x2[0:30] = x2[0:30] + 1
x2[31:60] = x2[31:60] - 1
y = [1 if i < 60 else 0 for i in range(0,100)]
df = pd.DataFrame({"x1": x1,"x2": x2,"y" : y})
X = df.iloc[:,[0,1]]
Y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)


# In[198]:


SVM = svm.SVC(kernel='poly', degree = 2)
SVM.fit(x_train,y_train)


# In[199]:


x_test = np.array(x_test)
y_test = np.array(y_test)


# In[200]:


plot_decision_regions(x_test, y_test, clf=SVM, legend=2)


# In[201]:


sns.scatterplot(x1, x2, hue = y)


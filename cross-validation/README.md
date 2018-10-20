# 交叉检验（Cross Validation)

This is translated from http://scikit-learn.org/stable/modules/cross_validation.html

本文翻自：http://scikit-learn.org/stable/modules/cross_validation.html

翻译：张亚东

Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally.

学习预测函数的参数，并且在同样的数据集上检验其表现是一个方法性的错误：
这个只是在已训练过的重复的样本集上表现得很好的模型，却会在输入未曾训练过的样本集时表现得很差。
这个现象就叫做过拟合现象。为了避免该问题，通常一个实用的方法就是从已有的数据集划分一部分作为测试集 ```X_test, y_test```，进而开展一个监督学习模型的试验。
需要注意的是，这个“试验”并不是仅仅为了学术研究，因为即使在商用机器学习模型当中，通常也必须进行试验。

In scikit-learn a random split into training and test sets can be quickly computed with the train_test_split helper function. Let’s load the iris data set to fit a linear support vector machine on it:

在scikit-learn当中，可以使用```train_test_split```函数快速实现训练集和测试集的随机划分。
我们来使用iris数据拟合一个线性支持向量机：

```python
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> from sklearn import datasets
>>> from sklearn import svm

>>> iris = datasets.load_iris()
>>> iris.data.shape, iris.target.shape
((150, 4), (150,))
```

We can now quickly sample a training set while holding out 40% of the data for testing (evaluating) our classifier:

我们拿出40%的数据用于检验我们的分类情况，同时可以快速得到训练集：

```python
>>> X_train, X_test, y_train, y_test = train_test_split(
...     iris.data, iris.target, test_size=0.4, random_state=0)

>>> X_train.shape, y_train.shape
((90, 4), (90,))
>>> X_test.shape, y_test.shape
((60, 4), (60,))

>>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
>>> clf.score(X_test, y_test)                           
0.96...
```

When evaluating different settings (“hyperparameters”) for estimators, such as the C setting that must be manually set for an SVM, there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called “validation set”: training proceeds on the training set, after which evaluation is done on the validation set, and when the experiment seems to be successful, final evaluation can be done on the test set.

当我们为模型评估不同的超参数设置，例如SVM当中的```C```必须手动调整，但尽管这样，仍然存在着在测试集上出现过拟合的风险，因为参数在优化模型过程中，可能需要被微调。
这样的话，测试集的信息可能会泄露进入模型当中，那么评价指标将不再反应模型的普遍表现。
为了解决这个问题，还可以将数据集的另一个部分分离出来，也就是通常被称为“检验集”：
当评估在检验集上完成后，开始在训练集上训练，并且当试验看起来比较成功的时候，最终的评估才会在测试集上进行。

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets.

然而，通过将已获得的数据分成三部分，我们大大地减少了可以用于模型学习的样本的数量，并且结果可能会依赖于训练和检验的某个特定的随机选择。

A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”:
A model is trained using  of the folds as training data;
the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).

该问题的一种解决方案被称为交叉检验，简称CV。
测试集仍然作为最终的评估而被分离出来，但是在CV过程中，我们不再需要检验集。
在k-fold CV方法中，训练集被划分为 _k_ 个小的集合（其他方式将会在下面介绍，但大致上都是基本方法的推广），
下面是对每个“fold”（褶皱）过程的介绍：

+ 用<img src="http://latex.codecogs.com/gif.latex?k-1" />个fold作为训练集训练模型
+ 然后训练后的模型用剩下的数据检验（通常作为一个测试集计算模型的表现/精度）

The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data (as is the case when fixing an arbitrary validation set), which is a major advantage in problems such as inverse inference where the number of samples is very small.

_k_ fold交叉检验的模型表现就是在循环当中的平均值。
这种方法在计算方面比较繁琐，但是没有浪费过多的数据，当样本量非常小的时候，（例如反推断问题）这个优点就尤为突出。

## 计算交叉检验度量 （Computing cross-validated metrics）

The simplest way to use cross-validation is to call the cross_val_score helper function on the estimator and the dataset.

使用交叉检验的最简单方法就是在评估器和数据集基础上调用```cross_val_score```函数。

The following example demonstrates how to estimate the accuracy of a linear kernel support vector machine on the iris dataset by splitting the data, fitting a model and computing the score 5 consecutive times (with different splits each time):

下面的例子基于iris数据集通过划分数据集、拟合模型以及计算5次连续的分数（每次使用不同的分隔方式)，展示了如何评估该线性核支持向量机：

```python
>>> from sklearn.model_selection import cross_val_score
>>> clf = svm.SVC(kernel='linear', C=1)
>>> scores = cross_val_score(clf, iris.data, iris.target, cv=5)
>>> scores                                              
array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])
```

The mean score and the 95% confidence interval of the score estimate are hence given by:

平均的分数以及95%置信区间的分数估计如下：

```python
>>> print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Accuracy: 0.98 (+/- 0.03)
```

By default, the score computed at each CV iteration is the score method of the estimator. It is possible to change this by using the scoring parameter:

在这里，每次CV计算的分数就是评估器的分数。这是可以通过```scoring```参数修改的。

```python
>>> from sklearn import metrics
>>> scores = cross_val_score(
...     clf, iris.data, iris.target, cv=5, scoring='f1_macro')
>>> scores                                              
array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])
```

See The scoring parameter: defining model evaluation rules for details. In the case of the Iris dataset, the samples are balanced across target classes hence the accuracy and the F1-score are almost equal.

详细参考[The scoring parameter: defining model evaluation rules for details](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)。
在Iris数据集案例当中，样本在目标类别当中是平等的，所以```score```和```F1-score```是几乎相等的。

When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default, the latter being used if the estimator derives from ClassifierMixin.



It is also possible to use other cross validation strategies by passing a cross validation iterator instead, for instance:


















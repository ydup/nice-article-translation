# 机器学习中的张量方法(Tensor Methods in Machine Learning)

This original blog is from https://www.offconvex.org/2015/12/17/tensor-decompositions/.

本文翻自https://www.offconvex.org/2015/12/17/tensor-decompositions/。

翻译： 张亚东

Tensors are high dimensional generalizations of matrices.
In recent years tensor decompositions were used to design learning algorithms for estimating parameters of latent variable models like Hidden Markov Model, Mixture of Gaussians and Latent Dirichlet Allocation (many of these works were considered as examples of “spectral learning”, read on to find out why). 
In this post I will briefly describe why tensors are useful in these settings.

张量是矩阵的高维推广。在近些年，张量分解被广泛应用于为那些具有潜在变量的模型设计学习算法，例如隐藏马尔科夫模型、高斯和隐藏Dirichlet分配的混合方法。通常，这些方法都被认为是光谱学习的实例。
在这篇文章，我将会简单介绍张量为什么在这些领域如此神通广大。

Using Singular Value Decomposition (SVD), we can write a matrix M∈Rn×m as the sum of many rank one matrices:

通过奇异值分解方法（SVD），我们可以把一个矩阵 <img src="http://latex.codecogs.com/gif.latex?\textbf{M}%20\in%20R_{n%20\times%20m}" />  写作很多个秩为1的矩阵和的形式：
<div align=center>
<img src="http://latex.codecogs.com/gif.latex?\textbf{M}=\sum_{i=1}^{r}{\lambda_i\overrightarrow{u_i}\overrightarrow{v_i}^{T}}" />
</div>

When the rank r is small, this gives a concise representation for the matrix M (using (m+n)r parameters instead of mn). Such decompositions are widely applied in machine learning.

如果矩阵的秩r比较小，那么就可以给出一个对矩阵M的简介的表达形式（使用了(m+n)r个参数，而不是mn个）。这样的分解在机器学习中广泛应用。

Tensor decomposition is a generalization of low rank matrix decomposition. Although most tensor problems are NP-hard in the worst case, several natural subcases of tensor decomposition can be solved in polynomial time. Later we will see that these subcases are still very powerful in learning latent variable models.

张量分解是低秩矩阵分解的推广。尽管大多数张量问题在最坏的情况下都是NP难问题，但是很多张量分解的子问题都可以在多项式级别的时间复杂度内解决。后面我们会看到，这些子问题在学习潜在变量模型当中仍旧是十分强大的。

## 矩阵分解 (Matrix Decompositions)

Before talking about tensors, let us first see an example of how matrix factorization can be used to learn latent variable models. In 1904, psychologist Charles Spearman tried to understand whether human intelligence is a composite of different types of measureable intelligence. Let’s describe a highly simplified version of his method, where the hypothesis is that there are exactly two kinds of intelligence: quantitative and verbal. Spearman’s method consisted of making his subjects take several different kinds of tests. Let’s name these tests Classics, Math, Music, etc. The subjects scores can be represented by a matrix M, which has one row per student, and one column per test.

在将张量之前，让我们先看矩阵分解如何可以用于学习隐藏变量的例子。
在1940年，心理学家Charles Spearman尝试理解人类的智力是否可以分解为多种衡量智力方式的类型。我们来看对Charles Spearman方法的一个高度简化的版本——假设人类的智力由两个部分组成，量化和语言。
Charles Spearman通过进行各种不同的考试来支持他的理论。考试科目包括：文学，数学，音乐等等。这些课程的分数用矩阵M来表示，其中，每一行代表一个学生，每一列代表一种科目。

The simplified version of Spearman’s hypothesis is that each student has different amounts of quantitative and verbal intelligence, say xquant and xverb respectively. Each test measures a different mix of intelligences, so say it gives a weighting yquant to quantitative and yverb to verbal. Intuitively, a student with higher strength on verbal intelligence should perform better on a test that has a high weight on verbal intelligence. Let’s describe this relationship as a simple bilinear function:

简化一下Charles Spearman的假说，每一个学生具有不同的量化水平和语言水平，各自用<img src="http://latex.codecogs.com/gif.latex?x_quant" />和<img src="http://latex.codecogs.com/gif.latex?x_verb}" /> 符号代表。
每一个科目的测试衡量了不同程度智力水平的组合，（例如数学科目当中衡量量化水平的程度要更高一些），所以给量化水平和语言水平不同的权重，分别用<img src="http://latex.codecogs.com/gif.latex?y_quant" />和<img src="http://latex.codecogs.com/gif.latex?y_verb}" /> 代表。
直观得说，一个学生如果拥有较高的语言水平（相对于量化水平），那么他/她应该可以在语言水平权重比较高的科目中获得更优异的成绩。（例如语言水平更高的同学的语文成绩应该要比数学成绩更高一些）。
那么我们用一个简单的二元线性函数来描述这个现象：

<div align=center>
<img src="http://latex.codecogs.com/gif.latex?score=x_quant%20\times%20y_quant+x_verb%20\times%20y_verb" />
</div>

Denoting by x verb,x quant the vectors describing the strengths of the students, and letting y verb,y quant be the vectors that describe the weighting of intelligences in the different tests, we can express matrix M as the sum of two rank 1 matrices (in other words, M has rank at most 2):

用<img src="http://latex.codecogs.com/gif.latex?\overrightarrow{x}_quant" />和<img src="http://latex.codecogs.com/gif.latex?\overrightarrow{x}_verb" /> 向量表示学生样本的两个智力水平，用<img src="http://latex.codecogs.com/gif.latex?\overrightarrow{y}_quant" />和<img src="http://latex.codecogs.com/gif.latex?\overrightarrow{y}_verb" />表示在不同考试当中两个智力类型的权重，我们可以用两个秩为1的矩阵和表示M，（而M的秩最大为2）：

<div align=center>
<img src="http://latex.codecogs.com/gif.latex?\textbf{M}=\overrightarrow{x}_quant%20\overrightarrow{y}^T_quant+\overrightarrow{x}_verb%20overrightarrow{y}^T_verb" />
</div>

Thus verifying that M has rank 2 (or that it is very close to a rank 2 matrix) should let us conclude that there are indeed two kinds of intelligence.

因此，确认的矩阵M的秩为2（或者说是很接近秩为2的矩阵，这里指的是矩阵在科目方向上的秩很接近2）可以让我们推断出确实只有这两种智力水平。


Note that this decomposition is not the Singular Value Decomposition (SVD). SVD requires strong orthogonality constraints (which translates to “different intelligences are completely uncorrelated”) that are not plausible in this setting.

学要说明的是，这里的分解并不是奇异值分解（SVD），SVD具有很强的正交性约束条件，换句话说就是不同的智力类型是完全不相关的，而在这个问题中正交约束并不合理。













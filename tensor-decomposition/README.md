# 机器学习当中的张量方法(Tensor Methods in Machine Learning)

This original blog is from https://www.offconvex.org/2015/12/17/tensor-decompositions/.

本文翻自https://www.offconvex.org/2015/12/17/tensor-decompositions/。

翻译： 张亚东

Tensors are high dimensional generalizations of matrices.
In recent years tensor decompositions were used to design learning algorithms for estimating parameters of latent variable models like Hidden Markov Model, Mixture of Gaussians and Latent Dirichlet Allocation (many of these works were considered as examples of “spectral learning”, read on to find out why). 
In this post I will briefly describe why tensors are useful in these settings.

张量是矩阵的高维推广。在近些年，张量分解被广泛应用于为那些具有潜在变量的模型设计学习算法，例如隐藏马尔科夫模型、高斯和隐藏Dirichlet分配的混合方法。通常，这些方法都被认为是光谱学习的实例。
在这篇文章，我将会简单介绍张量为什么在这些领域如此神通广大。

Using Singular Value Decomposition (SVD), we can write a matrix M∈Rn×m as the sum of many rank one matrices:

通过奇异值分解方法（SVD），我们可以把一个矩阵 <img src="http://latex.codecogs.com/gif.latex?M\inR_{n\timesm}" />  写作很多个秩为1的矩阵和的形式：

<img src="http://latex.codecogs.com/gif.latex?M=\sum_{i=1}^{r}{\lambda_i\overrightarrow{u_i}\overrightarrow{v_i}^{T}}" />

When the rank r is small, this gives a concise representation for the matrix M (using (m+n)r parameters instead of mn). Such decompositions are widely applied in machine learning.

如果矩阵的秩r比较小，那么就可以给出一个对矩阵M的简介的表达形式（使用了(m+n)r个参数，而不是mn个）。这样的分解在机器学习中广泛应用。

Tensor decomposition is a generalization of low rank matrix decomposition. Although most tensor problems are NP-hard in the worst case, several natural subcases of tensor decomposition can be solved in polynomial time. Later we will see that these subcases are still very powerful in learning latent variable models.

张量分解是低秩矩阵分解的推广。尽管大多数张量问题在最坏的情况下都是NP难问题，但是很多张量分解的子问题都可以在多项式级别的时间复杂度内解决。后面我们会看到，这些子问题在学习潜在变量模型当中仍旧是十分强大的。

## 矩阵分解 (Matrix Decompositions)

Before talking about tensors, let us first see an example of how matrix factorization can be used to learn latent variable models. In 1904, psychologist Charles Spearman tried to understand whether human intelligence is a composite of different types of measureable intelligence. Let’s describe a highly simplified version of his method, where the hypothesis is that there are exactly two kinds of intelligence: quantitative and verbal. Spearman’s method consisted of making his subjects take several different kinds of tests. Let’s name these tests Classics, Math, Music, etc. The subjects scores can be represented by a matrix M, which has one row per student, and one column per test.







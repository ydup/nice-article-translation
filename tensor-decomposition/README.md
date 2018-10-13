

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

通过奇异值分解方法（SVD），我们可以把一个矩阵 \\ M \in  R_{n \times m} \\  写作很多个秩为1的矩阵和的形式：

<img src="http://latex.codecogs.com/gif.latex?M=\sum_{i=1}^{r}{\lambda_i\overrightarrow{u_i}\overrightarrow{v_i}^{T}}" />
 ![img](http://latex.codecogs.com/gif.latex?M=\sum_{i=1}^{r}{\lambda_i\overrightarrow{u_i}\overrightarrow{v_i}^{T}})

$$  M = \sum_{i=1}^{r} { \lambda_i \overrightarrow{u_i} \overrightarrow{v_i}^{T} } $$

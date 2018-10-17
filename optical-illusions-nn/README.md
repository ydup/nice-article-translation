# 神经网络：什么是视觉幻想？我不懂(Neural networks don't understand what optical illusions are)

This original blog is from https://www.technologyreview.com/s/612261/neural-networks-dont-understand-what-optical-illusions-are/amp/.

本文翻自https://www.technologyreview.com/s/612261/neural-networks-dont-understand-what-optical-illusions-are/amp/。

Machine-vision systems can match humans at recognizing faces and can even create realistic synthetic faces. But researchers have discovered that the same systems cannot recognize optical illusions, which means they also can’t create new ones.

机器视觉系统可以在面部识别方面和人类媲美，并且甚至可以自己合成人脸。
但是研究者已经发现，这样的系统无法辨别视觉幻想，也就意味着，它们也无法生成新的幻象图。

Human vision is an extraordinary facility. Although it evolved in specific environments over many millions of years, it is capable of tasks that early visual systems never experienced. Reading is a good example, as is identifying artificial objects such as cars, planes, road signs, and so on.

人类视觉是一种非凡的能力。尽管在特定的环境中不断进化了数百万年，但是它可以完成早些视觉系统根本无法经历过的任务。
阅读就是一个很好的例子，同时还有辨识人造的物体，例如汽车、飞机、道路标识等等。

But the visual system also has a well-known set of shortcomings that we experience as optical illusions. Indeed, researchers have identified many ways in which these illusions cause humans to misjudge color, size, alignment, and movement.

但是视觉系统也存在广为人知的缺陷，也就是我们会被视觉欺骗。
事实上，研究者们已经找到了很多方式，可以让人类被幻觉欺骗而导致对颜色、大小、线和移动的判断失误。

The illusions themselves are interesting because they provide insight into the nature of the visual system and perception. So ways of finding new illusions that explore these limits would be hugely useful.

幻觉本身就很有意思，因为它们让我们可以进一步理解视觉系统和感官的本质。
所以发现新的突破现有局限的幻象就显得意义重大。

Which is where deep learning comes in. In recent years, machines have learned to recognize objects and faces in images and then to create similar images themselves. So it’s easy to imagine that a machine-vision system ought to be able to learn to recognize illusions and then to create its own.

深度学习将何去何从？在近些年，机器已经学会识别图像中的物体和面部并且可以去自行创造相似的图像。
所以很容易想象机器视觉系统应该也能够学着识别和创造幻象。

Enter Robert Williams and Roman Yampolskiy at the University of Louisville in Kentucky. These guys have attempted this feat but found that things aren’t so simple. Current machine-learning systems cannot generate their own optical illusions—at least not yet. Why not?

这些人尝试着去突破，但是发现事情并不简单。当前的机器学习系统无法（至少现在还不能）生成它们自己的视觉幻象。那么究竟是为什么呢？

First some background. The recent advances in deep learning are based on two advances. The first is the availability of powerful neural networks and one or two programming tricks that make them good at learning.

首先来了解一下背景。当前深度学习的不断进步的动力实际上基于以下两个方面。第一是越来越容易搭建强大的神经网络以及总有一些让它们擅于“学习”的编程技巧。

The second is the creation of huge annotated databases that machines can learn from. Teaching a machine to recognize faces, for example, requires many tens of thousands of images containing faces that are clearly labeled. With that information, a neural net can learn to spot characteristic facial patterns—two eyes, a nose, and a mouth, for example. And even more impressive, a pair of them—called a generative adversarial network—can teach each other to create realistic, but totally synthetic, images of faces.







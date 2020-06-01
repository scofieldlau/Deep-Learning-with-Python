# Chapter 1. What is deep learning?

## 1.1 Artificial intelligence, machine learning, and deep learning



Figure 1.1. Artificial intelligence, machine learning, and deep learning

![img](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Images/01fig01.jpg)

### 1.1.1 AI

*the effort to automate intellectual tasks normally performed by humans*. As such, AI is a general field that encompasses machine learning and deep learning, but that also includes many more approaches that don’t involve any learning.



##### *symbolic AI*  (1950s - 1980s)

For a fairly long time, many experts believed that human-level artificial intelligence could be achieved by having programmers handcraft a sufficiently large set of explicit rules for manipulating knowledge. This approach is known as ***symbolic AI***, and it was the dominant paradigm in AI from the 1950s to the late 1980s. It reached its peak popularity during the *expert systems* boom of the 1980s.

it turned out to be intractable to figure out explicit rules for solving more complex, fuzzy problems, such as image classification, speech recognition, and language translation. A new approach arose to take symbolic AI’s place: ***machine learning*.**



### 1.1.2 Machine Learning

Machine learning arises from this question: could a computer go beyond “what we know how to order it to perform” and learn on its own how to perform a specified task? Could a computer surprise us? Rather than programmers crafting data-processing rules by hand, could a computer automatically learn these rules by looking at data?

This question opens the door to a new programming paradigm. In classical programming, the paradigm of symbolic AI, humans input rules (a program) and data to be processed according to these rules, and out come answers (see [figure 1.2](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Text/01.xhtml#ch01fig02)). With machine learning, humans input data as well as the answers expected from the data, and out come the rules. These rules can then be applied to new data to produce original answers.

Figure 1.2. Machine learning: a new programming paradigm

![img](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Images/01fig02.jpg)

### 1.1.3. LEARNING REPRESENTATIONS FROM DATA

what machine--learning algorithms *do*.

- **Input data points—** For instance, if the task is speech recognition, these data points could be sound files of people speaking. If the task is image tagging, they could be pictures.
- **Examples of the expected output—** In a speech-recognition task, these could be human-generated transcripts of sound files. In an image task, expected outputs could be tags such as “dog,” “cat,” and so on.
- **A way to measure whether the algorithm is doing a good job—** This is necessary in order to determine the distance between the algorithm’s current output and its expected output. The measurement is used as a feedback signal to adjust the way the algorithm works. This adjustment step is what we call *learning*.



A machine-learning model transforms its input data into meaningful outputs, a process that is “learned” from exposure to known examples of inputs and outputs. Therefore, the central problem in machine learning and deep learning is to *meaningfully transform data*: in other words, to learn useful *representations* of the input data at hand—representations that get us closer to the expected output. 



Machine-learning models are all about finding appropriate representations for their input data—transformations of the data that make it more amenable to the task at hand, such as a classification task.



Let’s make this concrete. Consider an x-axis, a y-axis, and some points represented by their coordinates in the (x, y) system, as shown in [figure 1.3](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Text/01.xhtml#ch01fig03).

Figure 1.3. Some sample data

![img](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Images/01fig03.jpg)

As you can see, we have a few white points and a few black points. Let’s say we want to develop an algorithm that can take the coordinates (x, y) of a point and output whether that point is likely to be black or to be white. In this case,

- The inputs are the coordinates of our points.
- The expected outputs are the colors of our points.
- A way to measure whether our algorithm is doing a good job could be, for instance, the percentage of points that are being correctly classified.

So that’s what machine learning is, technically: searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal. This simple idea allows for solving a remarkably broad range of intellectual tasks, from speech recognition to autonomous car driving.



### 1.1.4. THE “DEEP” IN DEEP LEARNING

Deep learning is a specific subfield of machine learning: a new take on learning representations from data that puts an emphasis on learning successive *layers* of increasingly meaningful representations. The *deep* in *deep learning* stands for this idea of successive layers of representations. 

Modern deep learning often involves tens or even hundreds of successive layers of representations—and they’re all learned automatically from exposure to training data. Meanwhile, other approaches to machine learning tend to focus on learning only one or two layers of representations of the data; hence, they’re sometimes called *shallow learning*.



In deep learning, these layered representations are (almost always) learned via models called *neural networks*, structured in literal layers stacked on top of each other. The term *neural network* is a reference to neurobiology. **Deep learning is a mathematical framework for learning representations from data.**



Figure 1.5. A deep neural network for digit classification

![img](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Images/01fig05.jpg)



As you can see in [figure 1.6](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Text/01.xhtml#ch01fig06), the network transforms the digit image into representations that are increasingly different from the original image and increasingly informative about the final result. You can think of a deep network as a multistage information-distillation operation, where information goes through successive filters and comes out increasingly *purified* (that is, useful with regard to some task).

如图1.6所示，多层神经网络将数字图像转换为与原始图像越来越不同且对最终结果的信息越来越丰富的表示形式。您可以将深层网络视为多级信息蒸馏操作，其中信息经过连续的过滤器并逐渐被净化出来（这对某些任务很有用）。

Figure 1.6. Deep representations learned by a digit-classification model

![img](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Images/01fig06_alt.jpg)

So that’s what deep learning is, technically: a multistage way to learn data representations. It’s a simple idea—but, as it turns out, very simple mechanisms, sufficiently scaled, can end up looking like magic.



### 1.1.5. UNDERSTANDING HOW DEEP LEARNING WORKS, IN THREE FIGURES

The specification of what a layer does to its input data is stored in the layer’s *weights*, which in essence are a bunch of numbers. In technical terms, we’d say that the transformation implemented by a layer is *parameterized* by its weights (see [figure 1.7](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Text/01.xhtml#ch01fig07)). (Weights are also sometimes called the *parameters* of a layer.) 

In this context, *learning* means finding a set of values for the weights of all layers in a network, such that the network will correctly map example inputs to their associated targets. 

But here’s the thing: a deep neural network can contain tens of millions of parameters. Finding the correct value for all of them may seem like a daunting task, especially given that modifying the value of one parameter will affect the behavior of all the others!

Figure 1.7. A neural network is parameterized by its weights.

![img](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Images/01fig07.jpg)

To control the output of a neural network, you need to be able to measure how far this output is from what you expected. This is the job of the *loss function* of the network, also called the *objective function*. The loss function takes the predictions of the network and the true target (what you wanted the network to output) and computes a distance score, capturing how well the network has done on this specific example (see [figure 1.8](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Text/01.xhtml#ch01fig08)).

Figure 1.8. A loss function measures the quality of the network’s output.

![img](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Images/01fig08.jpg)

The fundamental trick in deep learning is to use this score as a feedback signal to adjust the value of the weights a little, in a direction that will lower the loss score for the current example (see [figure 1.9](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Text/01.xhtml#ch01fig09)). This adjustment is the job of the *optimizer*, which implements what’s called the *Backpropagation* algorithm: the central algorithm in deep learning. 

Figure 1.9. The loss score is used as a feedback signal to adjust the weights.

![img](https://learning.oreilly.com/library/view/deep-learning-with/9781617294433/OEBPS/Images/01fig09.jpg)

Initially, the weights of the network are assigned random values, so the network merely implements a series of random transformations. Naturally, its output is far from what it should ideally be, and the loss score is accordingly very high. But with every example the network processes, the weights are adjusted a little in the correct direction, and the loss score decreases. This is the ***training loop***, which, repeated a sufficient number of times (typically tens of iterations over thousands of examples), yields weight values that minimize the loss function. A network with a minimal loss is one for which the outputs are as close as they can be to the targets: a trained network. Once again, it’s a simple mechanism that, once scaled, ends up looking like magic.



### 1.1.6. WHAT DEEP LEARNING HAS ACHIEVED SO FAR

Although deep learning is a fairly old subfield of machine learning, it only rose to prominence in the early 2010s.

In particular, deep learning has achieved the following breakthroughs, all in historically difficult areas of machine learning:

- Near-human-level image classification
- Near-human-level speech recognition
- Near-human-level handwriting transcription
- Improved machine translation
- Improved text-to-speech conversion
- Digital assistants such as Google Now and Amazon Alexa
- Near-human-level autonomous driving
- Improved ad targeting, as used by Google, Baidu, and Bing
- Improved search results on the web
- Ability to answer natural-language questions
- Superhuman Go playing

我们仍在探索深度学习可以做什么的全部范围。我们已开始将其应用于机器感知和自然语言理解之外的各种问题，例如形式推理。如果成功的话，这可能预示着深度学习将协助人类进行科学，软件开发等工作的时代。

### 1.1.7. DON’T BELIEVE THE SHORT-TERM HYPE

Although deep learning has led to remarkable achievements in recent years, expectations for what the field will be able to achieve in the next decade tend to run much higher than what will likely be possible. Although some world-changing applications like autonomous cars are already within reach, many more are likely to remain elusive for a long time, such as believable dialogue systems, human-level machine translation across arbitrary languages, and human-level natural-language understanding. In particular, talk of *human-level general intelligence* shouldn’t be taken too seriously. The risk with high expectations for the short term is that, as technology fails to deliver, research investment will dry up, slowing progress for a long time.

We may be currently witnessing the third cycle of AI hype and disappointment—and we’re still in the phase of intense optimism. It’s best to moderate our expectations for the short term and make sure people less familiar with the technical side of the field have a clear idea of what deep learning can and can’t deliver.



### 1.1.8. THE PROMISE OF AI

Although we may have unrealistic short-term expectations for AI, the long-term picture is looking bright. 

Right now, it may seem hard to believe that AI could have a large impact on our world, because it isn’t yet widely deployed—much as, back in 1995, it would have been difficult to believe in the future impact of the internet. Back then, most people didn’t see how the internet was relevant to them and how it was going to change their lives. The same is true for deep learning and AI today. But make no mistake: AI is coming. In a not-so-distant future

Don’t believe the short-term hype, but do believe in the long-term vision. It may take a while for AI to be deployed to its true potential—a potential the full extent of which no one has yet dared to dream—but AI is coming, and it will transform our world in a fantastic way.
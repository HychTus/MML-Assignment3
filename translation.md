# Task 3-1: Image Captioning with RNNs 

- 在本任务中，您将实现RNN（A部分）/LSTM（B部分），并使用它们训练一个模型，生成图像的描述。
- 该任务包含两个部分，总分为45分。


# Section A: Vanilla Recurrent Neural Networks

在这一部分，总共有13个步骤，总分为30分

# (1) COCO 数据集
在此任务中，我们将使用 2014 版本的 [COCO 数据集](https://cocodataset.org/)，这是一个用于图像描述的标准测试集。该数据集包含 80,000 张训练图像和 40,000 张验证图像，每张图像都有 5 条由 Amazon Mechanical Turk 工作者编写的描述。 （您可以从 [百度云](https://pan.baidu.com/s/1kmcCuvfMUadqSAezx384Ng?pwd=mml3) 下载数据集，密码是 `mml3`。）

**图像特征。** 我们已经为您预处理了数据并提取了特征。对于所有图像，我们从在 ImageNet 上预训练的 VGG-16 网络的 fc7 层提取了特征，这些特征存储在 `train2014_vgg16_fc7.h5` 和 `val2014_vgg16_fc7.h5` 文件中。为了减少处理时间和内存需求，我们使用主成分分析（PCA）将特征的维度从 4096 降低到 512，这些特征存储在 `train2014_vgg16_fc7_pca.h5` 和 `val2014_vgg16_fc7_pca.h5` 文件中。原始图像占用了近 20GB 的空间，因此我们没有将它们包含在下载中。由于所有图像都来自 Flickr，我们将训练和验证图像的 URL 存储在 `train2014_urls.txt` 和 `val2014_urls.txt` 文件中，这样您就可以按需下载图像以进行可视化。

**描述。** 处理字符串效率较低，因此我们将使用编码版本的描述。每个单词都被分配一个整数 ID，使我们能够通过整数序列表示描述。整数 ID 与单词之间的映射关系存储在 `coco2014_vocab.json` 文件中，您可以使用 `mml/coco_utils.py` 文件中的 `decode_captions` 函数将整数 ID 的 NumPy 数组转换回字符串。

**标记。** 我们在词汇表中添加了几个特殊的标记，并已为您处理了所有与特殊标记相关的实现细节。我们在每个描述的开始和结束分别添加特殊的 `<START>` 和 `<END>` 标记。稀有单词被替换为特殊的 `<UNK>` 标记（表示“未知”）。此外，由于我们希望使用包含不同长度描述的迷你批次进行训练，我们在 `<END>` 标记后使用特殊的 `<NULL>` 标记填充短描述，并且不会为 `<NULL>` 标记计算损失或梯度。

您可以使用 `mml/coco_utils.py` 文件中的 `load_coco_data` 函数加载所有 COCO 数据（描述、特征、URL 和词汇表）。请运行以下代码单元来执行此操作：


### 检查数据

在开始处理数据之前，查看数据集中的一些示例总是一个好主意。

你可以使用 `mml/coco_utils.py` 文件中的 `sample_coco_minibatch` 函数，从 `load_coco_data` 返回的数据结构中采样小批量数据。运行以下代码来采样一小批训练数据，并显示图像及其对应的描述。多次运行并查看结果，有助于你更好地了解数据集。

# (2) 循环神经网络 (RNN) [5 分]

如讲座中所讨论的，我们将使用循环神经网络 (RNN) 语言模型进行图像描述。文件 `mml/rnn_layers.py` 包含了实现循环神经网络所需的不同层类型，而文件 `mml/classifiers/rnn.py` 使用这些层来实现图像描述模型。

我们将首先在 `mml/rnn_layers.py` 中实现不同类型的 RNN 层。

**注意：** 长短期记忆 (LSTM) RNN 是一种常见的 Vanilla RNN 变种。我们将在下一个任务 `3.2-LSTM_Captioning.ipynb` 中实现 LSTM。

# (3) Vanilla RNN：前向传播步骤 [3 分]

打开文件 `mml/rnn_layers.py`。该文件实现了循环神经网络中常用的不同层的前向和后向传播。

首先实现函数 `rnn_step_forward`，该函数实现了 Vanilla 循环神经网络单个时间步的前向传播。实现完成后，运行以下代码来检查你的实现。你应该看到误差在 e-8 或更小的数量级。



# (10) 时序 Softmax 损失

在 RNN 语言模型中，在每个时间步，我们为词汇表中的每个单词生成一个得分。我们知道每个时间步的真实标签，所以我们使用 Softmax 损失函数来计算每个时间步的损失和梯度。我们将时间步的损失求和并在小批量中进行平均。

然而，有一个问题：由于我们是在小批量上进行操作，并且不同的句子可能有不同的长度，我们将 `<NULL>` 标记附加到每个句子的末尾，以便它们都具有相同的长度。我们不希望这些 `<NULL>` 标记影响损失或梯度，因此除了得分和真实标签外，我们的损失函数还接受一个 `mask` 数组，该数组告诉它哪些得分元素应当计入损失。

我们已经为你实现了这个损失函数；你可以查看文件 `mml/rnn_layers.py` 中的 `temporal_softmax_loss` 函数。

运行以下单元格来进行损失的合理性检查，并对该函数进行数值梯度检查。你应该会看到梯度的误差应该在 `e-7` 或更小的范围内。


# (13) 测试时 RNN 采样 [4分]

与分类模型不同，图像描述模型在训练时和测试时的行为有很大的不同。在训练时，我们可以访问真实标签（ground-truth）描述，因此我们将真实标签的词作为输入传递给 RNN 的每个时间步。而在测试时，我们从词汇表的分布中进行采样，在每个时间步将采样的结果作为下一个时间步的输入传递给 RNN。

在文件 `mml/classifiers/rnn.py` 中，实现 `sample` 方法，用于测试时的采样。完成后，运行以下操作，在训练数据和验证数据上对你的过拟合模型进行采样。训练数据上的样本应该非常好，而验证数据上的样本可能不太有意义。


# (2) Transformer
正如你所见，RNN（循环神经网络）非常强大，但训练速度通常较慢。此外，RNN在编码长距离依赖关系时存在困难（尽管LSTM是缓解这一问题的一种方法）。2017年，Vaswani等人在他们的论文["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)中提出了Transformer模型，旨在：a) 引入并行性，b) 使模型能够学习长距离依赖关系。这篇论文不仅催生了自然语言处理领域中的著名模型如BERT和GPT，还引发了跨领域的广泛关注，包括计算机视觉。虽然我们在这里以图像字幕生成为背景介绍该模型，但注意力机制本身的概念要更为通用。


# 第二部分：使用CLIP和LM进行图像描述生成

**本部分共包含9个步骤，总分为40分。**

[CLIP](https://github.com/openai/CLIP)（对比语言-图像预训练）是由OpenAI提出的一种多模态预训练模型，结合了图像和文本模态，以实现强大的跨模态理解能力。

语言模型（LM）是自然语言处理（NLP）中的核心工具，旨在理解和生成符合语言规则的文本。它们以概率形式对语言序列进行建模，预测文本中单词或标记的可能性。

其中，[GPT-2](https://huggingface.co/openai-community/gpt2)（生成式预训练变换器2）是OpenAI于2019年发布的一种语言生成模型。它是GPT（生成式预训练变换器）系列的第二代，在文本生成、摘要和翻译等任务中表现出强大的能力。

## (1) 动机：
在本部分中，能否将**CLIP**的跨模态理解能力与语言模型的文本生成能力相结合，以实现既准确的图像理解又富有表现力的语言生成？
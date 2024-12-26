import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch

# 需要阅读整个 CaptioningSolver 类，了解其实现原理
#TODO: 阅读源码并且进行学习

"""
CaptioningSolver 封装了训练图像描述模型所需的所有逻辑
CaptioningSolver 使用在 `optim.py` 中定义的不同更新规则进行随机梯度下降

该求解器接受训练数据和验证数据及标签, 因此可以定期检查训练和验证数据上的分类准确性, 以防止过拟合

要训练模型, 您首先需要构造一个 CaptioningSolver 实例
将模型、数据集和各种选项（学习率、批量大小等）传递给构造函数
然后可以调用 `train()` 方法来运行优化过程并训练模型

`train()` 方法返回后, 模型的参数 (`model.params`) 将包含在训练过程中基于验证集表现最好的参数
此外, 实例变量 `solver.loss_history` 将包含训练过程中所有遇到的损失值
而 `solver.train_acc_history` 和 `solver.val_acc_history` 将分别包含每个周期 (epoch) 在训练集和验证集上的准确率

示例使用方式如下所示: 

```python
data = load_coco_data()
model = MyAwesomeModel(hidden_dim=100)
solver = CaptioningSolver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=100)
solver.train()
```

CaptioningSolver 依赖于符合以下 API 规范的模型对象: 
- model.params 必须是一个字典, 将字符串类型的参数名称映射到包含参数值的 numpy 数组
- model.loss(features, captions) 必须是一个函数, 计算训练时的损失和梯度，具有以下输入和输出:

  输入:
  - features: 一个数组，给定图像的一个小批量特征, 形状为 `(N, D)`。
  - captions: 一个数组，给定这些图像的描述, 形状为 `(N, T)`, 其中每个元素的值范围在 `(0, V]` 之间

  输出:
  - loss: 一个标量, 表示损失值。
  - grads: 一个字典, 键与 `self.params` 中的参数名称相同, 值为相应的参数梯度

对于 model 中定义的所有 params 都需要输出 grads
整个过程完全由 numpy 实现, 不涉及到 pytorch
"""

class CaptioningSolver(object):
    """
    A CaptioningSolver encapsulates all the logic necessary for training
    image captioning models. The CaptioningSolver performs stochastic gradient
    descent using different update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a CaptioningSolver instance,
    passing the model, dataset, and various options (learning rate, batch size,
    etc) to the constructor. You will then call the train() method to run the
    optimization procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists containing
    the accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A CaptioningSolver works on a model object that must conform to the following
    API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(features, captions) must be a function that computes
      training-time loss and gradients, with the following inputs and outputs:

      Inputs:
      - features: Array giving a minibatch of features for images, of shape (N, D
      - captions: Array of captions for those images, of shape (N, T) where
        each element is in the range (0, V].

      Returns:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data from load_coco_data

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
          rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        """


        """
        构造一个新的 CaptioningSolver 实例。

        必需的参数：
        - model: 一个符合上述 API 规范的模型对象
        - data: 一个包含训练和验证数据的字典, 来自 `load_coco_data`

        可选的参数：
        - update_rule: 一个字符串，指定在 `optim.py` 中使用的更新规则的名称, 默认值是 'sgd'
        - optim_config: 一个字典, 包含将传递给所选更新规则的超参数
            每个更新规则需要不同的超参数 (见 `optim.py`)
            但所有更新规则都需要一个 'learning_rate' 参数, 因此这个参数应始终存在
        - lr_decay: 一个标量，用于学习率衰减; 每个周期结束后, 学习率将乘以该值
        - batch_size: 用于在训练过程中计算损失和梯度的小批量大小
        - num_epochs: 训练的周期数
        - print_every: 整数, 每经过 `print_every` 次迭代, 就会打印一次训练损失
        - verbose: 布尔值; 如果设置为 `False`, 则在训练过程中不会打印任何输。
        """

        self.model = model
        self.data = data

        # Unpack keyword arguments
        # 通过关键字参数来进行传递, kwargs.pop() 方法用于删除字典中指定键的值并返回该值
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        # 检查是否有不合法的参数传入
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        # getattr() 将字符串转换为函数
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {} # 记录效果最好的参数
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
            # 对于 model 中不同的 params 分别设置不同的 optim_config
            # 前面的过程就相当于是 deep copy

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training 
        # 并不是训练所有数据, 而是每次都随机进行抽取
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        # Compute loss and gradient
        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    # 为什么 check_accuracy 没有实现, 也没有调用?
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
          much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """
        return 0.0

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        # 全部都转换成 iteration 计算

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )
                # % 的语法是格式化字符串, 这里的 %f 表示浮点数

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            # 每个 batch 在 step 中就已经进行了参数更新
            # 每个 epoch 时主要进行学习率衰减 (scheduler 和 optimizer 不是一个东西)
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

        # 整个过程中并没有进行 val accuarcy 的计算, 也没有进行参数的保留
        # caption 并不好直接通过正确率进行评估 (这个框架原本是用于分类的)

import tensorflow as tf
from inits import *
from metrics import *
from utils import *

flags = tf.compat.v1.flags  # 设置隐层
FLAGS = flags.FLAGS


_LAYER_UIDS = {}


def get_layer_uid(layer_name=""):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]




# 自定义层
class GraphConvolution(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_features_nonzero,
        ID = None,
        support=1,
        activation=tf.nn.relu,  # 激活层
        bias=False,
        kernel_init="glorot",
        dropout=0.0,  # 丢失值
        sparse_input=False,
        **kwargs
    ):
        super(GraphConvolution, self).__init__()
        allowed_kwargs = {"name"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, "Invalid keyword argument: " + kwarg
        ID = kwargs.get("name")
        if not ID:  # 如果未指明实例name则使用其类的名字+全局出现次数
            layer = self.__class__.__name__.lower()
            ID = layer + "_" + str(get_layer_uid(layer))
        self.ID = ID
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.support = support
        self.bias = bias
        self.kernel_init = kernel_init
        self.dropout = dropout
        self.sparse_input = sparse_input
        self.num_features_nonzero = num_features_nonzero
        
        if self.kernel_init == "glorot":
            self.weight = glorot_init([self.input_dim, self.output_dim], name=self.ID)
        else:
            self.weight = uniform_init(
                [self.input_dim, self.output_dim], name=self.ID
            )
        if self.bias:
            self.bias = zero_init([self.output_dim], name="bias")
        self.built = True

    def call(self, inputs,training=True):
        x, support = inputs
        if training :
            x = dropout(x,1 - self.dropout, self.num_features_nonzero, self.sparse_input) # dropout只有在训练的时候进行

        # 卷积操作
        tmp = dot(x, self.weight, sparse=self.sparse_input)
        output = dot(support, tmp, sparse=True)
        if self.bias:
            output += self.bias

        return self.activation(output)


class GCN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, num_features_nonzero, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features_nonzero = num_features_nonzero

        self.layers_ = []
        self.layers_.append(
            GraphConvolution(
                num_features_nonzero=self.num_features_nonzero,
                input_dim=self.input_dim,
                output_dim=FLAGS.hidden,
                sparse_input=True,
                dropout=FLAGS.dropout,
                kernel_init=FLAGS.kernel_init,
            )
        )

        self.layers_.append(
            GraphConvolution(
                num_features_nonzero=self.num_features_nonzero,
                activation=lambda x: x, # 自映射
                input_dim=FLAGS.hidden,
                output_dim=self.output_dim,
                dropout=FLAGS.dropout,
                kernel_init=FLAGS.kernel_init,
            )
        )

    def call(self, inputs):
        x, label, mask, support = inputs
        outputs = [x]
        for layer in self.layers_:
            hidden = layer((outputs[-1], support))
            outputs.append(hidden)
        output = outputs[-1]

        # Weight decay loss
        loss = tf.zeros([])
        for var in self.layers_[0].trainable_variables:
            loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # 计算损失函数和精度
        loss += masked_softmax_cross_entropy(output, label, mask)

        acc = masked_accuracy(output, label, mask)

        return loss, acc
    
    def predict(self):
        return tf.nn.softmax(self.outputs)

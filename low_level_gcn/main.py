from utils import *
from inits import *
from model import GCN, GraphConvolution
import time

# seed = 123
# np.random.seed(seed)
# tf.random.set_seed(seed)

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dataset", "cora", "Dataset string."
)  # 'cora'(140), 'citeseer'(120), 'pubmed'(60)
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("epochs", 200, "Number of epochs to train.")
flags.DEFINE_integer("hidden", 16, "Number of units in hidden layer 1.")
flags.DEFINE_float("dropout", 0.5, "Dropout rate (1 - keep probability).")
flags.DEFINE_float("weight_decay", 5e-4, "Weight for L2 loss on embedding matrix.")
flags.DEFINE_string("kernel_init", "glorot", "Initial weights method.")
flags.DEFINE_integer("train_set_size", 140, "Number of train set.")
flags.DEFINE_integer("repeat_num", 1, "Number of repetitions.")
flags.DEFINE_integer("early_stopping", 30, "Early stopping.")
test_acc_list = []
test_loss_list = []
t_test = time.time()
for _ in range(FLAGS.repeat_num): # 做重复实验
    print('------{}%------'.format((_)*100/FLAGS.repeat_num))
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_dataset(
        FLAGS.dataset,FLAGS.train_set_size
    )
    features = preprocess_features(features)
    support = preprocess_adj(adj)  # A-->\tilde(A)

    model_func = GCN
    model = model_func(
        input_dim=features[2][1],
        output_dim=y_train.shape[1],
        num_features_nonzero=features[1].shape,
    )

    train_label = tf.convert_to_tensor(y_train)
    train_mask = tf.convert_to_tensor(train_mask)
    val_label = tf.convert_to_tensor(y_val)
    val_mask = tf.convert_to_tensor(val_mask)
    test_label = tf.convert_to_tensor(y_test)
    test_mask = tf.convert_to_tensor(test_mask)
    features = tf.SparseTensor(*features)
    support = tf.cast(tf.SparseTensor(*support), dtype=tf.float32)
    num_features_nonzero = features.values.shape


    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    epoch_loss = list()
    for epoch in range(FLAGS.epochs):
        with tf.GradientTape() as tape:
            loss, acc = model((features, train_label, train_mask,support))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        val_loss, val_acc = model((features, val_label, val_mask, support),training = False)
        epoch_loss.append(val_loss)

        # Early stop
        if epoch > FLAGS.early_stopping and epoch_loss[-1] > np.mean(epoch_loss[-(FLAGS.early_stopping+1):-1]): 
            print("Early stopping..., epoch: {}".format(epoch))
            break
    test_loss,test_acc = model((features,test_label,test_mask,support),training = False)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

print("Test set results({} mean):".format(FLAGS.repeat_num), "cost=", "{:.5f}".format(np.array(test_loss_list).mean()),
"accuracy=", "{:.5f}".format(np.array(test_acc).mean()), "time=", "{:.5f}".format(time.time() - t_test))

    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(float(loss)),
    #     "train_acc=", "{:.5f}".format(float(acc)), "val_loss=", "{:.5f}".format(float(val_loss)),
    #     "val_acc=", "{:.5f}".format(val_acc))
    # test_loss,test_acc = model((features,test_label,test_mask,support),training = False)
    # print("Test set results:", "cost=", "{:.5f}".format(test_loss),
    # "accuracy=", "{:.5f}".format(test_acc))

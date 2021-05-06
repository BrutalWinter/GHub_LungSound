import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
#################################################

class Conv2D_loopbody(tf.keras.layers.Layer):
    # DarknetConv1D_BN_Leaky(32, 5, strides=(1,1))
    #       DarknetConv1D(*args, **self.no_bias_kwargs)
    def __init__(self, pool_size, *args, name="Conv2D_loopbody", **kwargs_outter):
        super(Conv2D_loopbody, self).__init__(name=name) ### initialize its parents
        self.kwargs = {}
        self.kwargs['use_bias'] = 'False'
        self.kwargs['kernel_initializer'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        # self.kwargs['padding'] = 'vaild' if kwargs_outter.get('strides') == 2 else 'same'
        self.kwargs['padding'] = 'same'
        # self.kwargs.update(kwargs_outter)

        self.Maxpool_kwargs = {}
        self.Maxpool_kwargs['strides'] = 2
        self.Maxpool_kwargs['data_format'] = None
        self.Maxpool_kwargs['padding'] = 'vaild' if kwargs_outter.get('half') == 1 else 'same'
        # self.Maxpool_kwargs['padding'] = 'same'

        self.conv=tf.keras.layers.Conv2D(*args, **self.kwargs)
        self.BN = tf.keras.layers.BatchNormalization()
        self.Relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.Pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, **self.Maxpool_kwargs)
        self.layer_Residual = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.BN(x)
        x = self.Relu(x)
        x = self.layer_Residual([inputs, x])
        return self.Pool(x)






class FC_Relu_layers(tf.keras.layers.Layer):
    def __init__(self, num_filters, name="FC_Relu_layers"):
        super(FC_Relu_layers, self).__init__(name=name)

        self.FC = tf.keras.layers.Dense(num_filters, activation=None)
        self.Relu = tf.keras.layers.LeakyReLU(alpha=0.1)


    def call(self, inputs):
        x = self.FC(inputs)
        x = self.Relu(x)

        return x




class LungSound_Model(tf.keras.Model):
    def __init__(self, outputs, kernelsize, name="Main_model"):
        super(LungSound_Model, self).__init__(name=name)

        self.layer0 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer1 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer2 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer3 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer4 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer5 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer6 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer7 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer8 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer9 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer10 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer11 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))
        self.layer12 =Conv2D_loopbody((5,5), outputs, kernelsize, strides=(1,1))

        self.flatten = tf.keras.layers.Flatten()
        self.Fc_Relu0 =FC_Relu_layers(num_filters=1024)
        self.Fc_Relu1 =FC_Relu_layers(num_filters=512)
        self.Fc_Relu2 =FC_Relu_layers(num_filters=256)

        self.Fc_end=tf.keras.layers.Dense(2, activation=None)
        self.Soft = tf.keras.layers.Softmax(axis=-1)




    def call(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        tf.print(x.shape)


        x1 = self.flatten(x)
        x1 = self.Fc_Relu0(x1)
        x1 = self.Fc_Relu1(x1)
        x1 = self.Fc_Relu2(x1)

        x1 = self.Fc_end(x1)
        x1 = self.Soft(x1)

        return x1


def Loss_calculator(label_pred, label_true):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    total_loss =scce(label_true,label_pred)
    return total_loss



##########################################################################
@tf.function
def train_step(Model,optimizer, x_batch_train, y_true):
    with tf.GradientTape() as tape:
        y_pred = Model(x_batch_train)
        total_loss = Loss_calculator(y_pred, y_true)

    grads = tape.gradient(total_loss, Model.trainable_weights)
    optimizer.apply_gradients(zip(grads, Model.trainable_weights))

    return total_loss


@tf.function
def predict(Model, x_batch_train):
  y_pred = Model(x_batch_train,training=False)
  return y_pred





if __name__ == '__main__':
    #############         model test:
    LungSound_model= LungSound_Model(32,5)

    # # Convert Keras model to ConcreteFunction
    # full_model = tf.function(lambda x: LungSound_model(x))
    # full_model = full_model.get_concrete_function(tf.TensorSpec(shape=[None,622,30,1], dtype=tf.float32, name="Input"))
    # # Get frozen ConcreteFunction
    # frozen_func = convert_variables_to_constants_v2(full_model)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-08)  # >minimum=1e-6
    # LungSound_model= Conv2D_loopbody(32,5, strides=(1,1))
    # print(LungSound_model.summary())
    ##############################################################
    data_set=tf.random.uniform(shape=[10,622,513,1], minval=0, maxval=10, dtype=tf.float32, seed=1)
    data_set= tf.data.Dataset.from_tensor_slices(data_set)
    True_label_set = tf.random.uniform(shape=[2,1], minval=0, maxval=1, dtype=tf.int32, seed=1)
    True_label_set = tf.data.Dataset.from_tensor_slices(True_label_set)
    dataset = tf.data.Dataset.zip((True_label_set, data_set))
    ##############################################################


    batch_size = 2
    dataset = dataset.shuffle(10).batch(batch_size)


    for epoch in tf.range(0, 10):
        for step, data in enumerate(dataset):
            PCM_batch = data[1]
            True_label= data[0]


            print('step',step)
            print('PCM_batch.shape',PCM_batch.shape)
            print('True_label',True_label.shape)
            # y_predict = predict(LungSound_model, PCM_batch)
            # print('Y_predict.shape', y_predict.shape)

            loss = train_step(LungSound_model, optimizer, PCM_batch, True_label)
            print('loss', loss)
            print('<===============================>')





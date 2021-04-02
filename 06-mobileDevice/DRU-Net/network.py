from tensorflow import keras


def drunet(channel):
    input = keras.layers.Input(shape=(None, None, channel), name='input')
    conv1 = residual_block(input, 32)
    pool1 = pool(conv1)
    conv2 = residual_block(pool1, 32)
    pool2 = pool(conv2)
    conv3 = residual_block(pool2, 32)
    pool3 = pool(conv3)
    conv4 = residual_block(pool3, 32)
    pool4 = pool(conv4)
    conv5 = residual_block(pool4, 32)

    deconv6 = deconv_layer(conv5, conv4, 32)
    deconv7 = deconv_layer(deconv6, conv3, 32)
    deconv8 = deconv_layer(deconv7, conv2, 32)
    deconv9 = deconv_layer(deconv8, conv1, 32)

    out = double_conv(deconv9,channel)
    out = keras.layers.Subtract()([input, out])
    model = keras.models.Model(inputs=input, outputs=out)
    return model


def double_conv(input, filters):
    conv1 = conv_layer(input, filters)
    conv2 = conv_layer(conv1, filters)
    return conv2

def pool(input):
    return keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(input)


def conv_layer(input, filters, bn=True, ac=True):
    out = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
                              padding='same', use_bias=False)(
        input)
    if bn:
        out = keras.layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=0.0001)(out)
    if ac:
        out = keras.layers.Activation(activation='relu')(out)
    return out


def deconv_layer(input, conv_prev, filter):
    up1 = keras.layers.UpSampling2D(size=(2, 2))(input)
    conv1 = keras.layers.Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
                                padding='same')(up1)
    add1 = conv_prev + conv1
    conv2 = residual_block(add1, filter)
    return conv2


def residual_block(layer, filters):
    for i in range(3):
        res = conv_layer(layer, filters)
        layer += res
    return layer


if __name__ == "__main__":
    model = drunet(1)
    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

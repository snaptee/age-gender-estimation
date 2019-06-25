from keras.layers import Dense, AveragePooling2D, Flatten, Dropout
from keras.applications import MobileNetV2
from keras.models import Model


def get_mobile_net(input_size, freeze_upper_weights=False):
    print('input_size', input_size)
    base = MobileNetV2(input_shape=(input_size, input_size, 3), include_top=False, weights='imagenet')
    drop = Dropout(1e-3)(base.output)
    if freeze_upper_weights:
        for layer in base.layers[:-10]:
            layer.trainable = False
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(drop)
    flatten = Flatten()(pool)
    age_layer = Dense(101, activation='softmax', name='pred_age')(flatten)

    model = Model(inputs=base.input, outputs=[age_layer])
    return model

from keras.applications import ResNet50, InceptionResNetV2, MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras.models import Model
from keras import backend as K


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_model(model_name="ResNet50"):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")

    prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction)

    return model


def get_mn_model(input_size):
    print('input_size', input_size)
    base = MobileNetV2(input_shape=(input_size, input_size, 3), include_top=False, weights='imagenet')
    #top_layer = GlobalAveragePooling2D()(base.output)
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(base.output)
    flatten = Flatten()(pool)
    gender_layer = Dense(2, activation='softmax', name='pred_gender')(flatten)
    age_layer = Dense(101, activation='softmax', name='pred_age')(flatten)

    model = Model(inputs=base.input, outputs=[gender_layer, age_layer])
    return model

def main():
    model = get_model("ResNet50")
    model.summary()


if __name__ == '__main__':
    main()

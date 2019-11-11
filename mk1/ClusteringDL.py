import os
import json
import click
import numpy as np
import tensorflow as tf


def create_model(shape):
    input_layer = tf.keras.Input(shape=(shape))
    layer = tf.keras.layers.Dense(64, activation="relu")(input_layer)
    layer = tf.keras.layers.Dense(128, activation="relu")(layer)
    layer = tf.keras.layers.Dense(256, activation="relu")(layer)
    layer = tf.keras.layers.Dense(512, activation="sigmoid")(layer)
    layer = tf.keras.layers.Dense(1024, activation="sigmoid")(layer)

    output_layer = tf.keras.layers.Dense(1024, activation="relu")(layer)
    output_layer = tf.keras.layers.Dense(512, activation="relu")(output_layer)
    output_layer = tf.keras.layers.Dense(256, activation="relu")(output_layer)
    output_layer = tf.keras.layers.Dense(128, activation="relu")(output_layer)
    output_layer = tf.keras.layers.Dense(64, activation="relu")(output_layer)
    output_layer = tf.keras.layers.Dense(32, activation="relu")(output_layer)
    output_layer = tf.keras.layers.Dense(16)(output_layer)
    output_layer = tf.keras.layers.Dense(8)(output_layer)
    output_layer = tf.keras.layers.Dense(4)(output_layer)
    output_layer = tf.keras.layers.Dense(2)(output_layer)
    output_layer = tf.keras.layers.Dense(1)(output_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def get_matrix(ds_location):
    with open(ds_location) as json_file:
        raw_list = json.load(json_file)
        data_matrix = []
        for data in raw_list:
            temp_row = []
            for column in data:
                temp_row.append(data[column])
            data_matrix.append(temp_row)
    return raw_list, np.asarray(data_matrix)


@click.command()
@click.option("--base_folder", "-b",
              default=r"D:\data_sets\linkedin-profiles-and-jobs-data\my",
              help="Base folder")
@click.option("--raw_json", "-r",
              default="_dump.json",
              help="Raw Input JSON")
def start(base_folder, raw_json):
    print("- Reading the RAW JSON")
    raw_list, data_matrix = get_matrix(os.path.join(base_folder, raw_json))
    train_matrix = data_matrix[:, 1:]

    print("- Creating the model")
    model = create_model(train_matrix.shape[1])

    print("- Training the model")
    model.compile(optimizer=tf.optimizers.SGD(0.01, 0.9), loss="kld")

    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1.0),
    #               loss="mean_squared_error",
    #               metrics=["mean_squared_error"])

    model.fit(train_matrix, train_matrix, batch_size=256, epochs=10)
    model.save(os.path.join(base_folder, "_temp_model.h5"))


if __name__ == '__main__':
    start()

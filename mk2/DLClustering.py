import click
import pandas as pd
import tensorflow as tf


def _create_model(input_shape):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu", input_shape=(input_shape, 1)))
    for x in range(5, 11):
        model.add(tf.keras.layers.Conv1D(2 ** x, kernel_size=1, activation="relu"))
    model.add(tf.keras.layers.Conv1D(1024, kernel_size=1, activation="sigmoid"))
    model.add(tf.keras.layers.Conv1D(1024, kernel_size=1, activation="sigmoid"))
    for x in range(10, 4, -1):
        model.add(tf.keras.layers.Conv1D(2 ** x, kernel_size=1, activation="relu"))
    for x in range(4, -1, -1):
        model.add(tf.keras.layers.Dense(2 ** x))

    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01, 0.9), loss="kld")
    return model


def _create_model_v2(input_shape):
    input_layer = tf.keras.layers.Input(shape=(input_shape, 1))

    x = tf.keras.layers.Conv1D(16, 2, activation="relu", padding="same")(input_layer)
    x = tf.keras.layers.MaxPooling1D(2, padding="same")(x)
    x = tf.keras.layers.Conv1D(32, 2, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling1D(2, padding="same")(x)
    x = tf.keras.layers.Conv1D(32, 2, activation="relu", padding="same")(x)
    encoded = tf.keras.layers.MaxPooling1D(2, padding="same")(x)

    x = tf.keras.layers.Conv1D(32, 2, activation="relu", padding="same")(encoded)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 2, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling1D(4)(x)
    x = tf.keras.layers.Conv1D(32, 2, activation="relu", padding="same")(x)
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.Dense(8)(x)
    x = tf.keras.layers.Dense(2)(x)
    decoded = tf.keras.layers.Dense(1)(x)

    autoencoder = tf.keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.SGD(0.01, 0.9), loss="kld")
    return autoencoder


def _create_model_v3(input_shape):
    input_layer = tf.keras.layers.Input(shape=(input_shape,))
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
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01, 0.9), loss="kld")
    return model


def _create_model_v4(input_shape):
    input_layer = tf.keras.layers.Input(shape=(input_shape,))
    layer = tf.keras.layers.Dense(64, activation="relu")(input_layer)
    layer = tf.keras.layers.Reshape((8, 8, 1))(layer)
    layer = tf.keras.layers.Conv2D(32, 2, activation="relu")(layer)
    layer = tf.keras.layers.Conv2D(64, 2, activation="relu")(layer)
    layer = tf.keras.layers.Conv2D(128, 2, activation="relu")(layer)
    layer = tf.keras.layers.Conv2D(256, 2, activation="relu")(layer)
    layer = tf.keras.layers.MaxPooling2D(2)(layer)
    layer = tf.keras.layers.Dense(1024, activation="relu")(layer)

    output_layer = tf.keras.layers.Dense(1024, activation="relu")(layer)
    output_layer = tf.keras.layers.Conv2D(256, 2, activation="relu")(output_layer)
    output_layer = tf.keras.layers.Conv2D(128, 1, activation="relu")(output_layer)
    output_layer = tf.keras.layers.Conv2D(64, 1, activation="relu")(output_layer)
    output_layer = tf.keras.layers.Conv2D(32, 1, activation="relu")(output_layer)
    output_layer = tf.keras.layers.MaxPooling2D(1)(output_layer)
    output_layer = tf.keras.layers.Reshape((32,))(output_layer)
    output_layer = tf.keras.layers.Dense(16)(output_layer)
    output_layer = tf.keras.layers.Dense(8)(output_layer)
    output_layer = tf.keras.layers.Dense(4)(output_layer)
    output_layer = tf.keras.layers.Dense(2)(output_layer)
    output_layer = tf.keras.layers.Dense(1)(output_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    # model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.8), loss="kullback_leibler_divergence")
    # model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.8), loss="kld")
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="kullback_leibler_divergence")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mean_squared_logarithmic_error")
    return model


@click.command()
@click.option("--input_csv", "-i",
              default=r"D:\data_sets\linkedin-profiles-and-jobs-data\dl\_dump.csv",
              help="Input CSV file")
@click.option("--output_csv", "-o",
              default=None,
              help="Output CSV file (If None, will gonna update input file)")
def start(input_csv, output_csv):
    raw_data = pd.read_csv(input_csv)
    data = raw_data[["ageEstimate",
                     "genderEstimate",
                     "startDate",
                     "endDate",
                     "avgMemberPosDuration",
                     "avgCompanyPosDuration",
                     "dateDuration",
                     "companyHasLogo",
                     "companyUrl",
                     "followable",
                     "hasPicture",
                     "isPremium",
                     "isFounderOrInvestor",
                     "isCXO",
                     "isManagerOrDirectorOrLead",
                     "isSoftwareArchitect",
                     "isArchitect",
                     "companyFollowerCount",
                     "companyStaffCount",
                     "connectionsCount",
                     "followersCount",
                     "positionId",
                     "companyNameId",
                     "mrbTitleAndPosTitleId"]]

    data = data.values
    data = data.reshape(data.shape[0], data.shape[1])

    model = _create_model_v4(data.shape[1])
    model.fit(data, data, batch_size=256, epochs=10)

    test = data[0].reshape(1, -1)
    pre = model.predict(test)
    print(test.shape)
    print(pre)
    print(pre.shape)


if __name__ == '__main__':
    start()

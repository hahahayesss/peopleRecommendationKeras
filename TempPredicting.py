import os
import json
import click
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


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


def _predict_all(model_location, matrix):
    model = tf.keras.models.load_model(model_location)
    model.summary()

    prediction_list = []
    for index, x in enumerate(model.predict(matrix)):
        prediction_list.append(x.astype(np.int64)[0])
    return prediction_list


def _normalize_matrix(prediction_list):
    min = prediction_list[0]
    max = prediction_list[0]
    for pre in prediction_list:
        if pre < min:
            min = pre
        if pre > max:
            max = pre

    for index, prediction in enumerate(prediction_list):
        prediction_list[index] = (prediction - min) / (max - min)
    return np.asanyarray(prediction_list).reshape(-1, 1)


def _find_best_k(data_matrix):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(4, 12), timings=False)
    visualizer.fit(data_matrix)
    visualizer.show()


@click.command()
@click.option("--base_folder", "-b",
              default=r"D:\data_sets\linkedin-profiles-and-jobs-data\my",
              help="Base folder")
@click.option("--raw_json", "-r",
              default="_dump.json",
              help="Raw Input JSON")
@click.option("--model_name", "-m",
              default="_temp_model.h5",
              help="Model location")
def start(base_folder, raw_json, model_name):
    print("- Reading the RAW JSON")
    raw_list, data_matrix = get_matrix(os.path.join(base_folder, raw_json))
    data_matrix = data_matrix[:, 1:]

    # temp = np.random.randint(0, data_matrix.shape[0], 50)
    # data_matrix = data_matrix[temp]

    # Data Prediction
    prediction_list = _predict_all(os.path.join(base_folder, model_name), data_matrix)
    normalized_list = _normalize_matrix(prediction_list)
    # / Data Prediction

    # Find the K value
    # _find_best_k(normalized_list)
    # / Find the K value

    # KMeans
    model = KMeans(n_clusters=6, init="k-means++")
    kmeans_prediction = model.fit_predict(normalized_list)
    kmeans_prediction = np.reshape(kmeans_prediction, (-1, 1))
    # / KMeans

    # Plot
    plt.scatter(normalized_list, kmeans_prediction)
    for x in model.cluster_centers_:
        plt.axvline(x=x, ymin=0, ymax=5, color="r")
    plt.show()
    # / Plot


if __name__ == '__main__':
    start()

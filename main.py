import numpy as np
import cv2
import os

from pathlib import Path
from random import randrange

from artificial_neural_network.ANN import ANN
from utils import metrics
from utils import plots

# Training information
NR_IMAGES = 1370
NR_CLASSES = 10
K_FOLDS = 8
OUTPUT_FOLDER = "output/model3/"

# Network hyper-parameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
LAYERS_DIMS = [IMAGE_HEIGHT*IMAGE_WIDTH*3, 200, 150, 100, 50, 20, 15, NR_CLASSES]
ITERS = 10000
LEARNING_RATE = 5e-3


def preprocess_image(src):
    dst = src.copy()

    # Normalize between 0 and 1
    dst = dst / 255.

    # Resize image
    dims = (IMAGE_WIDTH, IMAGE_HEIGHT)
    dst = cv2.resize(dst, dims, interpolation = cv2.INTER_NEAREST) 

    return dst


def get_data(source_folder):
    image_paths = Path(source_folder).rglob('*.jpg')

    data = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH * 3, NR_IMAGES))
    labels = np.zeros((NR_CLASSES, NR_IMAGES), dtype=int)
    indices = []

    for id_x, path in enumerate(image_paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        processed_img = preprocess_image(img)
        processed_img = processed_img.reshape(processed_img.shape[0] * processed_img.shape[1] * 3)
        data[:, id_x] = processed_img

        for k in range(NR_CLASSES):
            if f"n{k}" in str(path):
                labels[k, id_x] = 1
                indices.append(k)
                break
            

    return data, labels, np.array(indices)


def get_k_folds_cross_validation_data(x, y, indices, folds):
    """
        Balanced K-folds cross-validation data generation (one round of k folds).
        Balanced refers to the fact that the test set will contain a balanced number of data from each class.
    """

    length = x.shape[1]
    fold_size = length // folds

    nr_test_examples = length - fold_size * (folds - 1)
    nr_test_examples_per_class = nr_test_examples // NR_CLASSES
    nr_test_examples = nr_test_examples_per_class * NR_CLASSES
    nr_train_examples = length - nr_test_examples

    x_train = np.zeros((x.shape[0], nr_train_examples))
    y_train = np.zeros((NR_CLASSES, nr_train_examples), dtype=int)
    x_test = np.zeros((x.shape[0], nr_test_examples))
    y_test = np.zeros((NR_CLASSES, nr_test_examples), dtype=int)

    count_test_examples = 0
    count_train_examples = 0

    for j in range(NR_CLASSES):
        class_indices = np.where(indices == j)[0]
        np.random.shuffle(class_indices)
        test_indices = class_indices[0:nr_test_examples_per_class]
        train_indices = class_indices[nr_test_examples_per_class:]

        x_train[:, count_train_examples : count_train_examples + train_indices.shape[0]] = x[:, train_indices]
        x_test[:, count_test_examples : count_test_examples + test_indices.shape[0]] = x[:, test_indices]
        y_train[:, count_train_examples : count_train_examples + train_indices.shape[0]] = y[:, train_indices]
        y_test[:, count_test_examples : count_test_examples + test_indices.shape[0]] = y[:, test_indices]

        count_train_examples = count_train_examples + train_indices.shape[0]
        count_test_examples = count_test_examples + test_indices.shape[0]

    return x_train, y_train, x_test, y_test


def get_cross_validation_data(X, Y, cross_valid_split, fold_index):
    test_size = 0
    for split_indices in cross_valid_split:
        print(split_indices)
        test_size += split_indices[fold_index].shape[0]
    train_size = NR_IMAGES - test_size

    x_train = np.zeros((X.shape[0], train_size))
    y_train = np.zeros((Y.shape[0], train_size), dtype=int)
    x_test = np.zeros((X.shape[0], test_size))
    y_test = np.zeros((Y.shape[0], test_size), dtype=int)

    test_index = 0
    train_index = 0

    for i in range(NR_IMAGES):
        is_test = False
        for split_indices in cross_valid_split:
            if np.any(split_indices[fold_index] == i):
                is_test = True
                
        if is_test:
            x_test[:, test_index] = X[:, i]
            y_test[:, test_index] = Y[:, i]
            test_index += 1
        else:
            x_train[:, train_index] = X[:, i]
            y_train[:, train_index] = Y[:, i]
            train_index += 1
    
    return x_train, y_train, x_test, y_test
        

def get_cross_validation_split(indices):
    cross_validation_split = []

    for k in range(NR_CLASSES):
        class_indices = np.where(indices == k)

        split = np.array_split(class_indices[0], K_FOLDS)
        cross_validation_split.append(split)

    return cross_validation_split



if __name__ == "__main__": 
    # Read all image data and corresponding labels. 
    # X: image data, np array of shape (width * height * 3, nr_examples); Pre-processed using: normalization, resizing
    # Y: labels, np array of shape (nr_classes, nr_examples); stored in one-hot encoding format
    X, Y, indices = get_data("./monkeys_dataset/")
    print(indices)

    # Initialize performance metrics parameters
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    test_accuracy_arr = np.zeros(K_FOLDS)
    train_accuracy_arr = np.zeros(K_FOLDS)
    avg_confusion_matrix = np.zeros((NR_CLASSES, NR_CLASSES), dtype=int)
    avg_precision = np.zeros(NR_CLASSES)
    avg_recall = np.zeros(NR_CLASSES)
    avg_f_score = np.zeros(NR_CLASSES)

    # Write network parameters to file
    metrics_file = open(f"{OUTPUT_FOLDER}/metrics.txt", "w")
    metrics_file.write(f"Number of folds: {K_FOLDS}\n")
    metrics_file.write(f"Network layers: {LAYERS_DIMS}\n")
    metrics_file.write(f"Image width: {IMAGE_WIDTH}, image height: {IMAGE_HEIGHT}\n")
    metrics_file.write(f"Nr. iterations: {ITERS}\n")
    metrics_file.write(f"Learning rate: {LEARNING_RATE}\n")

    cross_validation_indices = get_cross_validation_split(indices)

    # K-fold cross-validation
    for k in range(K_FOLDS):
        print(f"\n\n\n FOLD {k} \n\n")
        metrics_file.write(f"\n\n  Fold {k} \n")

        # x_train, y_train, x_test, y_test = get_k_folds_cross_validation_data(X, Y, indices, K_FOLDS)
        x_train, y_train, x_test, y_test = get_cross_validation_data(X, Y, cross_validation_indices, k)

        # Fit data
        network = ANN(LAYERS_DIMS)
        train_losses, test_losses = network.fit(x_train, y_train, x_test, y_test, ITERS, LEARNING_RATE)

        # Plot train and test losses
        plots.plot_losses(train_losses, test_losses, K_FOLDS, f"{OUTPUT_FOLDER}/losses{k}")

        # Network prediction on test set
        y_pred, y_target = network.predict(x_test, y_test)

        # Compute test set metrics
        # Accuracy
        test_accuracy = metrics.get_accuracy(y_pred, y_target)
        test_accuracy_arr[k] = test_accuracy
        # Confusion matrix
        confusion_matrix = metrics.get_confusion_matrix(y_pred, y_target)
        avg_confusion_matrix = avg_confusion_matrix + confusion_matrix
        plots.plot_confusion_matrix(confusion_matrix, f"{OUTPUT_FOLDER}/confusion_matrix{k}")
        # Precision
        precision = metrics.get_precision(confusion_matrix)
        avg_precision += precision
        # Recall
        recall = metrics.get_recall(confusion_matrix)
        avg_recall += recall
        # F-score
        f_score = metrics.get_f_score(precision, recall) 
        avg_f_score += f_score
        # Plot precision, recall and f-score as table
        plots.plot_metrics_table(precision, recall, f_score, f"{OUTPUT_FOLDER}/metrics{k}")
        # Write metrics to file
        metrics_file.write(f"Accuracy: {test_accuracy}\n")
        metrics_file.write(f"Precision: {precision}\n")
        metrics_file.write(f"Recall: {recall}\n")
        metrics_file.write(f"F-score: {f_score}\n")

        # Compute train set accuracy
        y_pred, y_target = network.predict(x_train, y_train)
        train_accuracy = metrics.get_accuracy(y_pred, y_target)
        train_accuracy_arr[k] = train_accuracy

        print(f"Train accuracy: {train_accuracy}")
        print(f"Test accuracy: {test_accuracy}")

    # Average confusion matrix
    avg_confusion_matrix = avg_confusion_matrix / K_FOLDS
    avg_confusion_matrix = avg_confusion_matrix.astype(int)
    plots.plot_confusion_matrix(avg_confusion_matrix, f"{OUTPUT_FOLDER}/confusion_matrix")

    # Average precision
    avg_precision = np.around(avg_precision / K_FOLDS, 3)
    metrics_file.write(f"\n\nAverage precision: {avg_precision}\n")

    # Average recall
    avg_recall = np.around(avg_recall / K_FOLDS, 3)
    metrics_file.write(f"Average recall: {avg_recall}\n")

    # Average f-score
    avg_f_score = np.around(avg_f_score / K_FOLDS, 3)
    metrics_file.write(f"Average F-Score: {avg_f_score}\n")

    # Plot avg precision, recall and f-score as table
    plots.plot_metrics_table(avg_precision, avg_recall, avg_f_score, f"{OUTPUT_FOLDER}/metrics")

    # Compute mean and std of accuracies across folds
    metrics_file.write(f"\n All accuracies: {test_accuracy_arr}\n")
    mean_accuracy = round(np.mean(test_accuracy_arr), 3)
    metrics_file.write(f"Mean accuracy: {mean_accuracy}\n")
    std_accuracy = round(np.std(test_accuracy_arr), 3)
    metrics_file.write(f"Accuracy std: {std_accuracy}\n")

    print(f"\nAccuracy array: {test_accuracy_arr}")
    print(f"Mean accuracy: {mean_accuracy}")
    print(f"Accuracy std: {std_accuracy}")

    # Compute 95% confidence interval
    confidence_interval = metrics.get_confidence_interval(mean_accuracy, std_accuracy, K_FOLDS)
    metrics_file.write(f"95% confidence interval: {confidence_interval}")
    print(f"Confidence interval: {confidence_interval}")

    # Compute mean and std of train accuracies
    metrics_file.write(f"\n\n All train accuracies: {train_accuracy_arr}\n")
    mean_accuracy = round(np.mean(train_accuracy_arr), 3)
    metrics_file.write(f"Mean train accuracy: {mean_accuracy}\n")
    std_accuracy = round(np.std(train_accuracy_arr), 3)
    metrics_file.write(f"Train accuracy std: {std_accuracy}\n")

    print(f"\nTrain accuracy array: {train_accuracy_arr}")
    print(f"Mean train accuracy: {mean_accuracy}")
    print(f"Train accuracy std: {std_accuracy}")
    
    # Compute 95% confidence interval for train accuracies
    confidence_interval = metrics.get_confidence_interval(mean_accuracy, std_accuracy, K_FOLDS)
    metrics_file.write(f"Train accuracies 95% confidence interval: {confidence_interval}")
    print(f"Train accuracies confidence interval: {confidence_interval}")

    metrics_file.close()
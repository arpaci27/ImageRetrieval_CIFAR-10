import pickle
import numpy as np
import os
from matplotlib import pyplot as plt

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def extract_specific_classes(data, labels, classes):
    indices = [i for i, label in enumerate(labels) if label in classes]
    selected_data = data[indices]
    selected_labels = [labels[i] for i in indices]
    return selected_data, selected_labels

def convert_to_rgb(data):
    images = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    return images

def compute_histograms(images):
    histograms = []
    for img in images:
        hist_r = np.histogram(img[:,:,0], bins=256, range=(0,255))[0]
        hist_g = np.histogram(img[:,:,1], bins=256, range=(0,255))[0]
        hist_b = np.histogram(img[:,:,2], bins=256, range=(0,255))[0]
        histograms.append([hist_r, hist_g, hist_b])
    return histograms

# Paths to CIFAR-10 dataset
cifar10_dir = 'C:\\cifar-10-batches-py'  # Update with your CIFAR-10 directory path
batch_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

# Classes you're interested in (bird, cat, deer, dog, frog)
classes_of_interest = [2, 3, 4, 5, 6]  # CIFAR-10 labels for the classes

# Load and process the data
all_data, all_labels = [], []
for batch_name in batch_names:
    batch_path = os.path.join(cifar10_dir, batch_name)
    batch = load_cifar10_batch(batch_path)
    data, labels = batch[b'data'], batch[b'labels']
    selected_data, selected_labels = extract_specific_classes(data, labels, classes_of_interest)
    all_data.append(selected_data)
    all_labels.extend(selected_labels)

all_data = np.concatenate(all_data, axis=0)
rgb_images = convert_to_rgb(all_data)

histograms = compute_histograms(rgb_images)

plt.imshow(rgb_images[10])
plt.show()
plt.plot(histograms[0][0], color='red')
plt.plot(histograms[0][2], color='green')
plt.plot(histograms[0][2], color='blue')

def segregate_data(data, labels, class_label, train_count=40, test_count=5):
    indices = [i for i, label in enumerate(labels) if label == class_label]
    train_indices = indices[:train_count]
    test_indices = indices[train_count:train_count + test_count]
    return data[train_indices], data[test_indices]

# Segregate training and testing data for each class
training_data, testing_data = [], []
for class_label in classes_of_interest:
    train_data, test_data = segregate_data(rgb_images, all_labels, class_label)
    training_data.append(train_data)
    testing_data.append(test_data)

# Compute histograms for training data
training_histograms = [compute_histograms(data) for data in training_data]

# Compute histograms and perform tests for testing data
for i, test_images in enumerate(testing_data):
    test_histograms = compute_histograms(test_images)
    for test_hist in test_histograms:
        # Calculate Manhattan distances and find the 5 most similar images
        distances = []
        for train_hist in training_histograms[i]:
            distance = sum(abs(test_hist[channel] - train_hist[channel]).sum() for channel in range(3))
            distances.append(distance)
        closest_indices = np.argsort(distances)[:5]
        # closest_indices are the indices of the 5 most similar training images
        # You can use these indices to retrieve and analyze the corresponding images
def show_similar_images(test_image, similar_indices, training_data):
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 6, 1)
    plt.imshow(test_image)
    plt.title("Test Image")
    for i, index in enumerate(similar_indices, start=2):
        plt.subplot(1, 6, i)
        plt.imshow(training_data[index])
        plt.title(f"Similar {i-1}")
    plt.show()

# Example usage in your existing for loop
for i, test_images in enumerate(testing_data):
    test_histograms = compute_histograms(test_images)
    for j, test_hist in enumerate(test_histograms):
        distances = []
        for train_hist in training_histograms[i]:
            distance = sum(abs(test_hist[channel] - train_hist[channel]).sum() for channel in range(3))
            distances.append(distance)
        closest_indices = np.argsort(distances)[:5]
        show_similar_images(test_images[j], closest_indices, training_data[i])

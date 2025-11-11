import os
import cv2
import kagglehub
import numpy as np

# Download latest version
path = kagglehub.dataset_download("lexset/synthetic-asl-alphabet")

SIZE = 128
print("Path to dataset files:", path)


def load_dataset(subfolder):
    """Loads and normalizes all images from the given dataset subfolder."""
    print(f"Loading dataset for subfolder: {subfolder}")
    data, labels = [], []
    folder_path = os.path.join(path, subfolder)

    for letterfoldername in os.listdir(folder_path):
        letter_path = os.path.join(folder_path, letterfoldername)
        if not os.path.isdir(letter_path):
            continue

        print(f"Loading: {letterfoldername}")
        for file in os.listdir(letter_path):
            if not file.endswith(".png"):
                continue
            file_path = os.path.join(letter_path, file)
            img = cv2.imread(file_path)
            if img is None:
                continue
            img = cv2.resize(img, (SIZE, SIZE))
            img = img.astype(np.float32) / 255.0
            data.append(img)
            labels.append(letterfoldername)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)

    # Shuffle dataset
    p = np.random.permutation(len(data))
    return data[p], labels[p]


# Load both sets
training_data, training_labels = load_dataset("Train_Alphabet")
test_data, test_labels = load_dataset("Test_Alphabet")

print("Training:", training_data.shape, training_labels.shape)
print("Test:", test_data.shape, test_labels.shape)

# Save preprocessed data
np.savez("ASL_data", train_data = training_data, train_label = training_labels, test_data = test_data, test_label = test_labels)
# Potato Leaf Disease Classification (Early Blight, Healthy, Late Blight)

This repository showcases a **Potato Leaf Disease Classification** project. It uses a **transfer learning** approach with **MobileNetV2** to classify potato leaves into **Early Blight**, **Healthy**, or **Late Blight**. The dataset is taken from [Kaggle](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld/data).

> **Disclaimer**: This project is not a formal or production-grade solution. It is an **exploratory exercise** in collaboration with ChatGPT to demonstrate the capabilities of transfer learning on a plant disease dataset.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology & Code Explanation](#methodology--code-explanation)
4. [Results & Discussion](#results--discussion)
5. [Usage](#usage)
6. [Acknowledgments](#acknowledgments)
7. [Disclaimer](#disclaimer)

---

## Overview

Potato diseases, such as **Early Blight** and **Late Blight**, are critical concerns in agriculture. This project aims to **detect** and **classify** these diseases using **convolutional neural networks** (CNNs). By leveraging a **pretrained MobileNetV2** model, we can achieve relatively high accuracy even with a modest dataset.

### Key Objectives

- **Demonstrate Transfer Learning**: Use MobileNetV2 trained on ImageNet and fine-tune it for potato leaf classification.
- **Achieve Good Accuracy**: Obtain a classification accuracy of around **90%** on a three-class problem (Early Blight, Healthy, Late Blight).
- **Showcase Data Augmentation**: Use transformations such as rotation, shift, and zoom to improve generalization.

---

## Dataset

The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld/data). It consists of potato leaf images classified into three categories:
1. **Early_Blight**
2. **Healthy**
3. **Late_Blight**

**Directory Structure** (already partitioned):
```
Potato Disease Leaf Dataset
├── A. Testing
│   ├── Early_Blight
│   ├── Healthy
│   └── Late_Blight
├── B. Training
│   ├── Early_Blight
│   ├── Healthy
│   └── Late_Blight
└── C. Validation
    ├── Early_Blight
    ├── Healthy
    └── Late_Blight
```

- Each image is **256 × 256** pixels in `.jpg` format.
- The background is distinct, so no segmentation step was required.
- **Note**: For MobileNetV2, we typically resize images to **224 × 224** during the data loading process.

---

## Methodology & Code Explanation

### 1. **Data Loading and Augmentation**
We used **`ImageDataGenerator`** to:
- **Preprocess** images using MobileNetV2’s `preprocess_input`.
- **Augment** the training data (rotation, shifting, zooming, and flipping) to improve generalization.
- **Batch** images for efficient GPU utilization.

```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
```

### 2. **Transfer Learning with MobileNetV2**
We load **MobileNetV2** without the top layers (`include_top=False`) to use it as a **feature extractor**. The base is initialized with **ImageNet** weights.

```python
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')
```

### 3. **Custom Classification Head**
We add layers on top of the base model:
- A **GlobalAveragePooling2D** to condense spatial features.
- A **Dense(128)** + **Dropout(0.5)** for further feature learning and regularization.
- A final **Dense(3)** with **softmax** for multi-class output.

```python
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=outputs)
```

### 4. **Freezing & Fine-Tuning**
1. **Freeze** the base model layers and train only the new head for several epochs.
2. **Unfreeze** some top layers of the base model and fine-tune with a lower learning rate.

```python
# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile and train the new head
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Fine-tune with lower LR
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)
```

### 5. **Evaluation**
We then evaluate the model on the test set:

```python
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

## Results & Discussion

After **20 total epochs** (10 with the base frozen, 10 with partial unfreezing):
- **Test Accuracy** reached **~90%**.
- This is a significant improvement over a simple CNN trained from scratch, which might only reach ~50–60%.

**Why such improvement?**  
- **Transfer learning** leverages pretrained features from ImageNet, enabling the model to adapt quickly to leaf disease patterns.  
- **Data augmentation** helps prevent overfitting and improves generalization.

### Next Steps
- **Further Fine-Tuning**: Adjust the number of unfrozen layers and learning rates.
- **Experiment with Other Architectures**: e.g., ResNet50, EfficientNet, etc.
- **Additional Data**: More leaf images (or synthetic augmentation) could push accuracy even higher.

---

## Usage

1. **Clone or Download** this repository.
2. **Place the Kaggle Dataset** in the directory `Potato Disease Leaf Dataset`, maintaining the folder structure (`A. Testing`, `B. Training`, `C. Validation`).
3. **Run the Notebook** (e.g., in Google Colab):
   - Mount your Google Drive.
   - Install any missing dependencies (e.g., `tensorflow`, `keras`, `matplotlib`, `tqdm`).
   - Modify paths in the code as needed.
   - Execute cells to train the model and view results.

---

## Acknowledgments

- **Kaggle** for hosting the [Potato Disease Leaf Dataset](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld/data).  
- **TensorFlow/Keras** for providing user-friendly libraries for deep learning.  
- **ImageNet** and **MobileNetV2** authors for pretrained weights.  
- **ChatGPT** for co-creating and discussing the methodology.

---

## Disclaimer

This project is **not** a formal or production-grade solution. It is an **exploration** conducted in collaboration with **ChatGPT** to demonstrate the potential of transfer learning for plant disease classification. Accuracy metrics and conclusions are **for demonstration purposes only** and may not fully represent real-world performance.

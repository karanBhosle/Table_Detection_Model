# Table Detection and Recognition using CNN

## Project Overview

This project aims to develop a deep learning-based table detection and recognition system using a simple CNN model. The model is trained on the a subset of Original dataset: Pubtable1M dataset for table structure recognition, identifying table rows, columns, and spanning cells. Additionally, the system extracts tabular data from PDF documents, applies bounding box annotations, and outputs structured data in table format.

## Dataset

The project utilizes the Pubtable1M **Dataset**, which consists of:

- **Images** of tables with labeled structures such as rows, columns, and spanning cells.
- **Labels** indicating different table structures such as table rows, table columns, and spanning cells.

Each label file contains class annotations with bounding boxes corresponding to table elements.

## Implementation Steps

### 1. Data Preparation

A custom PyTorch dataset class (`TableDataset`) was created to:

- Load images and corresponding labels.
- Apply transformations such as resizing and conversion to tensors.
- Use one-hot encoding for multi-label classification.
- Handle dataset partitioning into training and validation sets.

**Transformations Applied:**

- Resize images to **128×128** (to optimize memory usage in Google Colab).
- Convert images to PyTorch tensors.

**Data Loaders:**

- `train_loader` and `val_loader` were implemented using the `DataLoader` API with a batch size of **16**.

### 2. CNN Model for Table Structure Recognition

A simple Convolutional Neural Network (CNN) was implemented with the following architecture:

- **Three convolutional layers** with ReLU activation and max-pooling.
- **Fully connected layers** to classify table structures into **three categories**.
- **Sigmoid activation** to support multi-label classification.
- **Binary Cross Entropy Loss (BCELoss)** as the loss function.
- **Adam optimizer** with a learning rate of **0.001**.

### 3. Training and Validation

The model was trained for **10 epochs** using the training set. The validation set was used to evaluate the performance after each epoch.

**Metrics Used:**

- **Loss** computation during training.
- **Validation accuracy** measured using binary classification metrics.
- **Precision and recall checks** for better evaluation.

### 4. Model Saving and Evaluation

- The trained model was saved using `torch.save(model.state_dict(), 'table_cnn.pth')`.
- After training, the model was evaluated on **five test images** to validate its accuracy.
- Random test data was used to check model predictions with bounding boxes.

### 5. PDF Table Extraction with Bounding Boxes

The project included a **PDF processing pipeline**:

- Convert PDF pages into images using `pdf2image.convert_from_path()`.
- Apply **EasyOCR** for text extraction with bounding box detection.
- Use the trained model to classify table structures.
- Draw bounding boxes on extracted tables for visualization.
- Present extracted table data in **pandas DataFrame format**.

### 6. Visualization and Output

- Extracted text with bounding boxes is displayed using **matplotlib**.
- Bounding boxes are drawn using OpenCV (`cv2.polylines()`).
- Extracted content is displayed in a structured **pandas DataFrame** format within the Jupyter notebook cell.

## Final Output

For each PDF page:

1. **Annotated Image** with detected table structure and bounding boxes.
2. **Extracted Table Content** displayed in a tabular format with bounding box coordinates, extracted text, and confidence scores.

**Example Output Table:**

| Bounding Box              | Extracted Text | Confidence |
| ------------------------- | -------------- | ---------- |
| [(x1, y1), (x2, y2), ...] | "Table Header" | 0.98       |
| [(x1, y1), (x2, y2), ...] | "Row Content"  | 0.96       |

## Conclusion

This project successfully implements a **deep learning-based table detection and recognition system** using a simple CNN model. The system:

- Detects table rows, columns, and spanning cells.
- Extracts tabular data from PDF documents.
- Displays bounding box annotations and structured table output.

Future improvements could include:

- Enhancing model accuracy with a deeper CNN or Transformer-based model.
- Using larger datasets for better generalization.
- Adding support for complex table structures such as merged cells.

This implementation provides a robust pipeline for **automated table extraction and recognition** from scanned or digital PDF documents.

This project was developed by Karan Bhosle. Connect on LinkedIn: [Karan Bhosle](https://www.linkedin.com/in/karanbhosle/).

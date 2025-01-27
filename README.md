# U-Net Skin Lesion Segmentation

This project implements a U-Net model, enhanced with layer normalization, for the task of skin lesion segmentation. The project includes the model definition in PyTorch and a training notebook for dataset preparation, model training, and evaluation.

---

## Features
- **U-Net Architecture**: Based on the original U-Net paper, with modifications to include layer normalization for improved convergence.
- **Training and Evaluation**: Provided Jupyter notebook to train the model on the ISIC 2018 dataset.
- **Customizability**: Easy to adjust model parameters, loss functions, and learning rate.

---

## Prerequisites
Ensure you have the following installed:

- Python 3.8 or later
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- Additional requirements listed in `requirements.txt`:
  - torch
  - Pillow
  - torchvision
  - scikit-learn
  - numpy
  - tensorboard
  - TensorFlow
  - albumentations

Install dependencies:
```bash
pip install -r "../requirements.txt"
```

---

## File Structure

- **`model.py`**: Contains the U-Net model definition.
- **`train.ipynb`**: A Jupyter notebook for training and evaluating the U-Net model.
- **`requirements.txt`**: A file listing all the dependencies for the project.

---

## Usage

1. **Model Definition**
   The U-Net model is implemented in `model.py`. It is designed to work with input images and predict pixel-level segmentation masks. You can modify parameters such as the number of classes in the model initialization, or the dimension of the input image.

2. **Training the Model**
   Use the `train.ipynb` notebook for:
   - Loading and preprocessing the dataset.
   - Training the U-Net model.
   - Visualizing the training progress using metrics and loss plots.

   Open the notebook:
   ```bash
   jupyter notebook train.ipynb
   ```

3. **Evaluation**
   The notebook includes cells to evaluate the model on a validation/test set and visualize predicted masks.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/lanceHEN/U-Net-ISIC-2018.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model using the Jupyter notebook:
   ```bash
   jupyter notebook train.ipynb
   ```

---

## Results
During training, you can monitor:
- **Loss curves**
- **Validation accuracy**
- **Predicted masks**

---

## Acknowledgments
- Original U-Net paper: [https://arxiv.org/pdf/1505.04597](https://arxiv.org/pdf/1505.04597)
- ISIC 2018 dataset for skin lesion segmentation

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.


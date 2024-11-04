# Plant Classification

![Sample Images from Each Class](assets/_dataset-samples.png)

This project provides a solution for classifying plants into 30 classes using ResNet and MobileNetV2 models.

* ResNet
  * https://arxiv.org/abs/1512.03385
  * https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
* MobileNetV2
  * https://arxiv.org/abs/1801.04381
  * https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

The models were fine-tuned based on the torchvision implementation and pretrained weights. Gradio is used for building a web interface, and Weights & Biases for experiments tracking.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ваш-проект/plant-classifier.git
    cd plant-classifier
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the dataset (for training):
   ```bash
   python data/get_data.py
   ```

5. Download pre-trained ImageNet model weights (for training):
   ```bash
   cd weights
   bash download_pretrained.sh
   ```

6. Download checkpoints:

   During training, the best weights of the model are saved based on validation performance.
   ```bash
   cd weights
   bash download_checkpoints.sh
   ```

## Usage

### Training the Model
   To train a model, specify either **mobilenet** or **resnet** using the `--model` argument.
   ```bash
   python src/train.py --model mobilenet
   python src/train.py --model resnet
   ```

   You can also adjust other parameters, such as the number of epochs, batch size, and learning rate, by adding additional arguments. For example:
   ```bash
   python src/train.py --model resnet --num-epochs 20 --batch-size 64 --learning-rate 0.001
   ```

### Launching the Gradio Interface
   ```bash
   python app.py
   ```
   Once the interface is running, you can select a model (ResNet or MobileNetV2), upload an image of a plant, and get the top three predicted classes and its probabilities.

   ![Spaces_screen](assets/spaces_screen.jpg)

   https://huggingface.co/spaces/eksemyashkina/plants-classification

## Results

| Model      | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|------------|------------|----------------|-----------|---------------|
| ResNet     | 0.13493    | 0.93058        | 0.28589   | 0.91233       |
| MobileNet  | 0.22018    | 0.90438        | 0.31522   | 0.8965        |


### Training Results of ResNet

![ResNet](assets/_resnet-plots.png)

The ResNet model was trained for 10 epochs on the plant classification dataset, with the following key metrics observed in the final epoch:

- **Train Loss**: 0.13493
- **Train Accuracy**: 0.93058
- **Test Loss**: 0.28589
- **Test Accuracy**: 0.91233

#### Observations from Training Curves

- **Training Accuracy**:
- The training accuracy curve shows a steady increase, with rapid initial improvement that gradually levels off near 0.93.
- This high training accuracy indicates that the model is effectively capturing the underlying patterns in the training data.

- **Training Loss**:
- The training loss decreases sharply at first, then tapers off as training progresses, stabilizing around 0.13.
- This steady decline in loss suggests that the model is converging well without signs of significant overfitting.

- **Validation Accuracy**:
- The validation accuracy improves consistently over the epochs, reaching 0.91233 in the final epoch.
- The validation accuracy closely matches the training accuracy, indicating good generalization performance.

- **Validation Loss**:
- The validation loss shows a consistent downward trend, ending at 0.28589 by the 10th epoch.
- The decreasing validation loss, alongside the high accuracy, suggests that the model is improving on unseen data without overfitting.

#### Summary

The ResNet model demonstrates excellent performance on the plant classification task, achieving a high test accuracy of 91.23% by the 10th epoch. The close alignment between training and validation metrics (both accuracy and loss) indicates that the model generalizes well, with minimal overfitting.


### Training Results of MobileNet

![MobileNet](assets/_mobilenet-plots.png)

The MobileNet model was trained for 10 epochs on the plant classification dataset, with the following key metrics observed in the final epoch:

- **Train Loss**: 0.22018
- **Train Accuracy**: 0.90438
- **Test Loss**: 0.31522
- **Test Accuracy**: 0.8965

#### Observations from Training Curves

- **Training Accuracy**:
- The training accuracy curve shows a strong initial increase, with accuracy rapidly improving before stabilizing near 0.90.
- This indicates that the model is effectively learning patterns in the training data.

- **Training Loss**:
- The training loss decreases sharply at first, then gradually stabilizes around 0.22.
- The steady decline in loss suggests that the model is converging well.

- **Validation Accuracy**:
- The validation accuracy improves consistently over the epochs, reaching 0.8965 in the final epoch.
- This value is slightly lower than the training accuracy, but the close match suggests good generalization performance without significant overfitting.

- **Validation Loss**:
- The validation loss shows a consistent downward trend, ending at 0.31522 by the 10th epoch.
- The decreasing validation loss, alongside the high validation accuracy, indicates that the model is performing well on unseen data.

#### Summary

The MobileNet model demonstrates solid performance on the plant classification task, achieving a test accuracy of 89.65% by the 10th epoch. The close alignment between training and validation metrics (both accuracy and loss) suggests that the model generalizes well to the validation data, with minimal overfitting.

Comparing these results to the MobileNet model, ResNet achieves slightly higher accuracy and lower test loss, suggesting that it may be more effective for this particular classification task. However, this improved performance may come at the cost of higher computational requirements, which should be considered depending on the deployment environment.
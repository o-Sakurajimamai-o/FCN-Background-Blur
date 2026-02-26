# FCN Background Blur

Data is [here](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset/data)

This project implements an automatic background blur effect (Portrait Mode) for images containing human subjects. 

It uses a Fully Convolutional Network (FCN) based on a ResNet-18 backbone to perform pixel-level human segmentation. The generated mask is then used to keep the human subject in sharp focus while applying a heavy Gaussian blur to the background.

##  How It Works

1. **Feature Extraction:** A pre-trained ResNet-18 model extracts image features.
2. **Segmentation:** A transposed convolution layer upsamples the features to generate a binary mask (Person vs. Background).
3. **Background Blurring:** The inference script (`pred.py`) uses `cv2.GaussianBlur` on the original image and blends it with the sharp subject using the predicted mask.

##  Project Structure

    ├── fcn.py                    # Training script and model definition
    ├── fcn_super_dataset.py      # Dataloader for the Supervisely Person Dataset
    ├── fcn_voc_dataset.py        # Dataloader for the Pascal VOC dataset (alternative)
    ├── pred.py                   # Inference script for the background blur effect
    ├── my_fcn2_model.pth         # Trained model weights (git-ignored)
    └── README.md                 

##  Dataset

The model is trained on the **Supervisely Person Dataset** to accurately distinguish human contours from complex backgrounds.
* Default dataset path in the code is set to: `D:\data\archive\supervisely_person_clean_2667_img`

##  Requirements

    pip install torch torchvision d2l
    pip install opencv-python matplotlib numpy Pillow

##  Usage

### 1. Apply Background Blur to Your Photo
If you already have the trained weights (`my_fcn2_model.pth`), you can directly run the inference script to blur the background of a custom image.

    python pred.py

*Note: Open `pred.py` and modify the `test_img` variable to the path of your image. The script handles resizing, mask prediction, and applying the blur effect.*

### 2. Train the Model (Optional)
To train the segmentation model from scratch:

    python fcn.py

*The training script uses SGD optimizer and Pixel-wise Cross-Entropy Loss. The final weights will be saved to your local directory.*
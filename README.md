# Urban scene segmentation with U-Net using OpenImages
**Author:** Katarina Koiro <br>
**LSP:** ...

## Task description
To create an image segmentation model, which classifies pixels into 3 or more classes.
Evaluate accuracy, precision, recall, and F1 on 100 unseen OpenImages images.
Additionally: benchmark against pretrained Segment Anything Model (SAM).

## Structure
There are two notebooks - `image_segmentation_data_prep.ipynb` dedicated for data exploration (class selection and validation, etc.) and data preparation (sampling with the selected classes, downloading images and instance masks, creation of dense masks, etc.) and `image_segmentation_training_eval.ipynb`, which is dedicated for model training and evaluation.

## Dataset: OpenImages

### More on the dataset and version choice:

OpenImages has been expanding annotations across versions (source: https://storage.googleapis.com/openimages/web/2022-10-25-announcing-v7-featuring-point-labels.html; https://storage.googleapis.com/openimages/web/factsfigures_v7.html):

- **V4 (2018):** 16M bounding boxes across 600 object classes on 1.9M images
- **V5 (2019):** Added **2.8M dense segmentation masks** across 350 classes -
  pixel-perfect binary masks marking object boundaries (this one's relevant for this task)
- **V6 (2020):** Added 675K localized narratives (multimodal image descriptions)
- **V7 (2022):** Added 66.4M **point-level labels** across 5,827 classes-
  sparse pixel annotations suitable for zero/few-shot segmentation (good but one main caveat regarding sparse points - they're designed for newer training approaches that can learn from sparse supervision rather than dense masks, such as dense segmentation masks)

**Choice of the version: V5 segmentation masks.**

Each version builds on top of the previous —> later versions do not replace
earlier annotations, they add new annotation types alongside them. As of my knowledge (checked via official website and various forums, blogs, etc), V7 did not update the segmentation masks since V5 release, moreover, the V5 segmentation data is stored as simple CSV files and ZIP archives on Google Cloud Storage, making it easy to directly download only the masks needed by filtering the CSV (especially useful for potential class exploration).

***Note for further consideration:*** considering the classes in this task are related to the urban/traffic scene, perhaps a better choice of a dataset would be Cityscapes Dataset (https://www.cityscapes-dataset.com/)

## Segmentation mask format

OpenImages provides **instance segmentation masks**, which is one binary PNG per object
instance, covering only the bounding box region of that instance. Non-zero pixels
indicate the object, zero pixels indicate background.

The annotation CSV files (`*-annotations-object-segmentation.csv`) contain one row
per instance mask with the following fields:
- **MaskPath**: filename of the PNG mask image
- **ImageID**: the image the mask belongs to
- **LabelName**: class identifier in MID format (e.g. `/m/01g317` for Person)
- **BoxXMin/XMax/YMin/YMax**: normalized bounding box coordinates of the starting
  box from which the mask was annotated
- **PredictedIoU**: machine-generated quality estimate of the mask
- **Clicks**: the annotator's guidance clicks used during the interactive
  segmentation process

For this semantic segmentation task, all instance masks are combined for a given image
into a single dense semantic mask where each pixel is assigned one class ID (0–5).
Overlapping instances of different classes are resolved by painting smaller objects
on top of larger ones, preserving the visibility of smaller classes like Traffic light.

## Chosen classes
| ID | Class | Colour |
|----|-------|-------|
| 0 | Background | Black |
| 1 | Person | Red |
| 2 | Car | Blue |
| 3 | Bus | Yellow |
| 4 | Traffic lights | Green |
| 5 | Skyscraper | Pink |

## Model
Custom U-Net with:
- **Encoder:** ResNet34 pretrained on ImageNet (for feature extraction)
- **Decoder:** Built from scratch (upsampling + skip connections + convolutions)
- **Output:** 1×1 convolution —> 6-class pixel-wise prediction

***Note: encoder provides strong learned features, while the decoder is entirely custom-built to perform semantic segmentation —> combining low-level spatial detail from skip connections with high-level semantic features from the encoder.***
## Loss
Combined Dice + Weighted Cross-Entropy for handling class imbalance.

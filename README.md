# Urban scene segmentation with U-Net using OpenImages

## Task description
To create an image segmentation model, which classifies pixels into 3 or more classes.
Evaluate accuracy, precision, recall, and F1 on 100 unseen OpenImages images.
Additionally: benchmark against pretrained Segment Anything Model (SAM).

## Structure

| Notebook | Its purpose |
|----------|---------|
| `01_data_preparation.ipynb` | Class exploration, sampling, downloading images/masks, dense mask creation |
| `02_model_training.ipynb` | Model training, evaluation, SAM benchmark |

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
   
### Training

**Two-phase strategy:**
1. **Phase 1 (frozen encoder):** Train decoder only, LR=1e-3, 10 epochs
2. **Phase 2 (fine-tuning):** Unfreeze encoder, differential LR (encoder: 5e-6, decoder: 1e-4)

**Loss:** Combined Dice + Weighted Cross-Entropy
- Class weights computed from inverse pixel frequency
- Person class weight boosted 1.5x to address class imbalance

**Regularization:**
- Dropout (0.1) in decoder blocks
- Early stopping (patience=5)
- Weight decay (1e-4)

**Image augmentations:** Horizontal flip, brightness, contrast, saturation, random scaling (all with 50% probability)

## Results

| Metric | Value |
|--------|-------|
| Pixel Accuracy | 0.811 |
| Macro F1 (excl. background) | 0.588 |
| Macro IoU (excl. background) | 0.432 |

**Per-class performance:**

| Class | Precision | Recall | F1 | IoU |
|-------|-----------|--------|-----|-----|
| Background | 0.988 | 0.787 | 0.876 | 0.780 |
| Person | 0.254 | 0.859 | 0.392 | 0.244 |
| Car | 0.514 | 0.908 | 0.656 | 0.488 |
| Bus | 0.717 | 0.942 | 0.814 | 0.687 |
| Traffic light | 0.444 | 0.927 | 0.600 | 0.429 |
| Skyscraper | 0.318 | 0.941 | 0.475 | 0.312 |

**Comment regarding metrics:** High recall (0.92) with lower precision (0.45) could be partially attributable to incomplete ground truth annotations in OpenImages, considering that the model correctly segments objects that were not labeled.

## SAM (Segment Anything Model) comparison

| | This model | SAM (ViT-B) |
|---|-----------|-------------|
| Inference time | 0.27s - 0.33s | 313s - 320s |
| Output | 6 semantic classes | 80-90 instance masks |
| Class labels | ✓ | ✗ |
| Speedup | **950x faster** | — |

SAM produces instance masks without semantic labels. SAM achieves much cleaner boundary detail, this task-specific model provides direct class predictions at a fraction of the computational cost, which is more suitable for real-time applications.

## Inference / model usage

```bash
# install dependencies
pip install -r requirements.txt
```

### Model weights

Weights not included due to size. To use pretrained weights:
1. Download from https://drive.google.com/file/d/1kkUF_0pURXze-1QKRJFwJUW2Bd-tRDcO/view?usp=sharing
2. Or retrain using the notebooks

To load, run:
```python
checkpoint = torch.load("best_model_v2.pt", map_location="cpu", weights_only=False)
model = UNetResNet34(num_classes=6, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```
after downloading weights, run:

```bash
# basic usage
python architecture_inference.py --image test_images/test_1.jpg

# specify checkpoint and device
python architecture_inference.py --image photo.jpg --checkpoint best_model_v2.pt --device cuda

# save output instead of displaying
python architecture_inference.py --image photo.jpg --output result.png
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | required | Path to input image |
| `--checkpoint` | `best_model_v2.pt` | Path to model weights |
| `--device` | auto-detect | `cuda` or `cpu` |
| `--output` | None | Save path (displays if not set) |

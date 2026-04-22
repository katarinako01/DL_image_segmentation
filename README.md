# Urban scene segmentation with U-Net using OpenImages

## Task description
To create an image segmentation model, which classifies pixels into 3 or more classes.
Evaluate accuracy, precision, recall and F1 on 100 unseen OpenImages images.
Additionally: benchmark against pretrained segmentation model, such as Segment Anything Model (SAM).

## Structure

| Item | Its purpose |
|----------|---------|
| `image_segmentation_data_prep.ipynb` | Class exploration, sampling, downloading images/masks, dense mask creation |
| `image_segmentation_training_eval.ipynb` | Model training, evaluation |
| `SegFormer_CLIP_sota_comparison.ipynb` | Exploration of model's saturation point by comparison to SegFormer and CLIPSeg |
| `architecture_inference.py` | Standalone inference script with model architecture |
| `test_images/` | Sample urban images for testing and demonstration |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

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

**Comment regarding metrics:** High recall with lower precision could be partially attributable to incomplete ground truth annotations in OpenImages, considering that the model correctly segments objects that were not labeled. More on annotation inconsistencies in OpenImages in this Towards Data Science article: https://towardsdatascience.com/i-performed-error-analysis-on-open-images-and-now-i-have-trust-issues-89080e03ba09/ 

## Model comparison & saturation analysis

To assess whether this model is approaching performance saturation, it was compared against two external baselines, i.e. SegFormer and CLIPSeg.

### Comparison models

**SegFormer** (Xie et al., NeurIPS 2021): A transformer-based state-of-the-art segmentation model pretrained on Cityscapes. Represents the best supervised approach currently available (that is specific to traffic scenes).

**CLIPSeg** (Lüddecke & Ecker, CVPR 2022): A zero-shot segmentation model that uses only text prompts ("person", "car", "bus", etc.) instead of pixel-level annotations. Built on CLIP's vision-language pretraining on 400 million image-text pairs.

### Results

#### SegFormer comparison (4 classes, excluding Skyscraper*)

| Model | Macro Precision | Macro Recall | Macro IoU |
|-------|-----------------|--------------|-----------|
| This model (U-Net ResNet34) | 0.484 | 0.911 | 0.465 |
| SegFormer | 0.662 | 0.744 | 0.485 |

*Skyscraper was excluded because mapping it to Cityscapes' broader "building" class resulted in unfair penalisation of SegFormer predictions.

#### CLIPSeg comparison (5 classes)

| Model | Macro IoU |
|-------|-----------|
| This model (U-Net ResNet34) | 0.433 |
| CLIPSeg (zero-shot) | 0.517 |

### Insights / findings

- All three have somewhat similar overall performance. All achieve IoU in the 0.43–0.52 range, despite fundamental differences in architecture and training approach.

- Shared failure pattern. All models struggle with the Person class (~0.25 IoU) and perform best on Bus (~0.68–0.77 IoU). This consistency across approaches suggests dataset-level challenges/issues rather than model-specific limitations (though they still are present, just in a way that, for example, changing architecture will give marginal improvement).

- Precision-recall trade-offs. This model shows high recall (0.911) with lower precision, indicating a tendency to over-segment. SegFormer and CLIPSeg are more conservative with higher precision but lower recall.

- Text-based approach is competitive. CLIPSeg achieves the highest IoU using only text prompts as supervision, showing that explicit pixel annotations might not even be necessary when strong vision-language pretraining is available.

### Interpretation

The convergence of three fundamentally different approaches to similar accuracy levels suggests that performance is constrained by annotation quality rather than model architecture. Further architectural improvements (such as attention mechanisms or deeper decoders) were tested but did not yield meaningful gains, reinforcing this conclusion.

## Inference / model usage

### Install dependencies

```bash
# install dependencies
pip install -r requirements.txt
```

### Model weights

Weights not included due to size. To use pretrained weights:
1. Download from [Google Drive](https://drive.google.com/file/d/1kkUF_0pURXze-1QKRJFwJUW2Bd-tRDcO/view?usp=sharing)
2. Or retrain using the notebooks

### Running inference

**Command line:**
```bash
# basic usage
python architecture_inference.py --image test_images/test_1.jpg --checkpoint best_model_v2.pt

# specify device
python architecture_inference.py --image photo.jpg --checkpoint best_model_v2.pt --device cuda

# save output instead of displaying
python architecture_inference.py --image photo.jpg --checkpoint best_model_v2.pt --output result.png
```

**In Python/Jupyter:**
```python
from architecture_inference import load_model, predict, visualize

model = load_model("best_model_v2.pt")
pred, img = predict(model, "photo.jpg")
visualize(img, pred)
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | required | Path to input image |
| `--checkpoint` | `best_model_v2.pt` | Path to model weights |
| `--device` | auto-detect | `cuda` or `cpu` |
| `--output` | None | Save path (displays if not set) |

## References

**Papers:**
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) (He et al., 2016)
- [Segment Anything Model (SAM)](https://arxiv.org/abs/2304.02643) (Kirillov et al., 2023)
- [Generalised Dice Overlap as a Deep Learning Loss Function](https://arxiv.org/abs/1707.03237) (Sudre et al., 2017)

**Dataset:**
- [OpenImages V5/V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- [OpenImages Annotation Error Analysis](https://towardsdatascience.com/i-performed-error-analysis-on-open-images-and-now-i-have-trust-issues-89080e03ba09/)

**Implementation references:**
- [U-Net Biomedical Image Segmentation](https://github.com/sauravmishra1710/U-Net---Biomedical-Image-Segmentation) — PyTorch U-Net implementation
- [SMP Encoder Comparison](https://smp.readthedocs.io/en/latest/encoders.html)
- [PyTorch ImageNet Normalization](https://pytorch.org/vision/stable/models.html)

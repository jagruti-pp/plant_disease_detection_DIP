# LeafVision – Plant Leaf Disease Detection
**Digital Image Processing Semester Mini Project**

**Team Members:** Samiksha Chavan, Jagruti Thotam

## Project Objective
Build a website that takes a plant-leaf photo through a complete Digital Image Processing (DIP) pipeline and then uses a trained image classifier to report whether the leaf is healthy or diseased, with the suspected affected region highlighted on a final annotated image.

## DIP Pipeline
```
Image Acquisition -> Enhancement -> Filtering -> Edge Detection -> Thresholding
-> Morphology -> Disease Detection -> Final Result
```

| Stage | Techniques shown on the website |
|---|---|
| Acquisition | Original image, width/height, channels, min/max intensity |
| Enhancement | Image negative, contrast stretching, gamma transformation |
| Filtering | Gaussian, Median, Laplacian, High-boost |
| Edge detection | Roberts, Prewitt, Sobel, Laplacian |
| Thresholding | Global, Otsu, Adaptive |
| Morphology | Erosion, Dilation, Opening, Closing |
| **Disease detection** | Trained Keras classifier + DIP-based localisation of suspicious regions |
| Final result | Annotated image, status, disease name, confidence, disease information, download button |

## Automatic Filter Selection (shown on every step)
Each stage measures the image it receives, **chooses** a technique, shows **why**, and passes the chosen result on:

| Step | What is measured | Choice rule |
|---|---|---|
| Enhancement | intensity range (p1-p99), mean brightness | narrow range -> Contrast Stretching; too dark/bright -> Gamma (γ computed from the mean); otherwise safe Contrast Stretching. Negative is never chosen (destroys colour cues). |
| Filtering | noise σ (Immerkaer), % outlier pixels, Laplacian variance | outliers > 1% -> Median; σ > 4 -> Gaussian; clean but soft -> High-boost; clean and sharp -> light Gaussian. Laplacian is an edge operator, not a filter. |
| Edge detection | edge-to-background ratio of each operator on the filtered image | highest ratio wins (Roberts / Prewitt / Sobel / Laplacian). |
| Thresholding | Otsu class separability, lighting drift inside each class | uneven light or weak two-peak histogram -> Adaptive; otherwise Otsu. Fixed T=120 is shown for comparison. |
| Morphology | number of noise specks and small holes in the mask | specks -> Opening; holes -> Closing; both/neither -> Opening then Closing. Erosion/Dilation shown but not chosen alone. |

On the website press **Next step** (or **Walk through step by step**) to move from one step to the next. The thresholds above are simple heuristics chosen for this project. The classifier's own pre-processing (contrast stretch + Gaussian) is unchanged so it still matches how the model was trained.

## Project Structure
```
plant_leaf_disease_project/
    app.py              Flask server, prediction, website training endpoints
    trainer.py          Model training code (used by website + train_model.py)
    dip_pipeline.py     All DIP operations (shared by app and training)
    disease_info.py     General symptoms / management text per disease
    train_model.py      Optional command-line training
    requirements.txt
    templates/index.html
    static/style.css, static/app.js
    model/              disease_model.keras + labels.json (created by training)
    training_data/      created automatically when you upload training images
    uploads/, outputs/
    sample_images/
```
There is no dataset folder to prepare: images are uploaded on the website.

## Installation
Python 3.9 - 3.12 is recommended (TensorFlow must support your Python version).
```
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## How to Run
```
python app.py
```
Open http://127.0.0.1:5000

## How to Train the Disease Model (on the website)
1. In the **Train the Disease Classifier** section, add images:
   - **Option A:** *Select Dataset Folder* - choose a folder with one sub-folder per class
     (e.g. `Healthy`, `Grape_Black_Rot`, `Grape_Esca`). Folder names become the class names.
   - **Option B:** type a class name, then *Choose Images & Add*.
2. You need at least 2 classes (include a class whose name contains **Healthy**). 20+ images per class is recommended.
3. Press **Train Model** and wait for the progress bar (several minutes on a normal laptop).
4. When it finishes, the model is loaded automatically - upload a leaf above to get the disease result.

Training uses transfer learning (MobileNetV2, ImageNet weights downloaded on first run, so internet is needed once; without internet it trains from scratch with lower accuracy and the page tells you). 20% of your images are held out to measure validation accuracy.

### Training from the command line (optional)
Put class folders inside `training_data/` and run:
```
python train_model.py
python train_model.py --epochs 12 --fine-tune-epochs 6
```
This is better for very large datasets (thousands of images), where uploading through the browser would be slow.

## How to Add an Already Trained Model
Copy these into the `model/` folder and restart `python app.py`:
```
model/disease_model.keras
model/labels.json        {"classes": ["Healthy", "Grape_Black_Rot", ...], "img_size": 224}
```
The order of `classes` must match the model's output order. The model must accept 0-255 RGB input (contain its own rescaling layer, as the one made by `trainer.py` does).
If the files are missing the website shows **"Disease classification model not available. Please train/load the model."** and does not report any disease.

## How Disease Detection Works
1. The DIP stages (enhancement, filtering, edge detection, thresholding, morphology) are computed and displayed.
2. The leaf is contrast-stretched and Gaussian-filtered (DIP pre-processing), resized, and passed to the trained classifier.
3. The **classifier alone** decides the disease name and confidence (softmax probability of the top class).
4. If the class is not "Healthy", colour segmentation (HSV analysis) + morphological opening/closing locate suspicious non-green / dark / whitish regions inside the leaf mask. These are outlined in red on the final image, with the percentage of leaf area covered.
5. If the class is Healthy, the leaf is shown unmarked as "Healthy Leaf".
6. General symptoms and management advice are shown. No medicine names or dosages are given.

## Limitations
- Accuracy depends entirely on the images you train with (quality, quantity, variety); leaves photographed in very different conditions from the training data may be misclassified. A low-confidence warning appears below 60%.
- The classifier only knows the classes present in the training dataset and will always pick one of them, even for an unrelated image.
- The red affected-region marking is a colour-based DIP estimate; it shows *where* the leaf looks abnormal, not a confirmed diagnosis, and can include shadows or natural discolouration.
- Leaf segmentation works best on a plain background.
- Disease information is general text; it is not a treatment plan.

> **This is an academic project and the result should be verified by an agricultural expert before treatment.**

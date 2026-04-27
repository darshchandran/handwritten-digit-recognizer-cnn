# Handwritten Digit Recognizer

I've always been curious about how computers "see" things. Building a neural network that can look at a handwritten number and figure out what digit it is. Sounds simple, but getting there involves understanding convolutional layers, pooling, dropout, and a bunch of other things that suddenly make a lot more sense when you're actually building something.

---

## What it does

Takes a 28x28 grayscale image of a handwritten digit as input and predicts which digit it is (0–9), along with a confidence score.

```
Input:  [Image of handwritten 7]
Output: Predicted digit → 7  (Confidence: 99.2%)

Input:  [Image of handwritten 3]
Output: Predicted digit → 3  (Confidence: 98.7%)
```

---

## Why I built it

MNIST is called the "hello world" of deep learning for a reason — it's the standard first project when you're learning CNNs. But beyond the cliché, it genuinely teaches you the full pipeline: preprocessing image data, designing a network architecture, understanding what each layer actually does, reading training curves, and interpreting a confusion matrix. All fundamentals that carry into every computer vision project after this.

---

## Dataset

**MNIST — Modified National Institute of Standards and Technology**  
70,000 grayscale images of handwritten digits (60,000 training / 10,000 test).  
Each image is 28x28 pixels. 10 classes — digits 0 through 9.  
Loaded directly via Keras: `keras.datasets.mnist.load_data()`

---

## Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — building and training the CNN
- **NumPy** — array operations and data reshaping
- **Matplotlib** — visualizing sample images, training curves, confusion matrix
- **Jupyter Notebook** — development environment

---

## How it works

### 1. Load & Explore
Pulled the dataset straight from Keras. Visualized a grid of sample images to get a feel for the data — handwriting varies a lot across 60,000 people.

### 2. Preprocess
- Normalized pixel values from 0–255 → 0–1 (helps the network train faster and more stably)
- Reshaped images from `(28, 28)` to `(28, 28, 1)` — the extra dimension tells Keras it's a grayscale image
- One-hot encoded labels: digit 3 becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

### 3. Build the CNN

```
Input → (28, 28, 1)

Conv2D (32 filters, 3x3) + ReLU      → detects basic edges and patterns
MaxPooling2D (2x2)                    → reduces size, keeps important features

Conv2D (64 filters, 3x3) + ReLU      → detects more complex shapes
MaxPooling2D (2x2)

Flatten                               → converts 2D feature maps to 1D vector
Dense (128 units) + ReLU             → fully connected layer
Dropout (0.5)                         → randomly drops neurons to prevent overfitting
Dense (10 units) + Softmax           → outputs a probability for each digit class
```

### 4. Compile & Train
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs:** 10, Batch size: 32
- Watched accuracy climb from ~87% at epoch 1 to ~99.4% by epoch 10

### 5. Evaluate
- Plotted training vs validation accuracy and loss curves over all 10 epochs
- Generated a 10x10 confusion matrix to see exactly where the model makes mistakes

### 6. Custom prediction
Tested on images drawn in MS Paint — saves as PNG, feeds into the model, prints prediction.

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 99.1% |
| Test Loss | 0.031 |
| Training Accuracy | 99.4% |

**Training progression:**
```
Epoch 1  → 87.3%
Epoch 3  → 96.1%
Epoch 5  → 98.4%
Epoch 10 → 99.4%
```

**Where it gets confused (from confusion matrix):**
- 4 occasionally misread as 9
- 3 occasionally misread as 5
- 7 occasionally misread as 1

These are honestly mistakes humans make too when handwriting is sloppy.

---

## How to run it

```bash
# Clone the repo
git clone https://github.com/yourusername/mnist-digit-recognizer.git
cd mnist-digit-recognizer

# Install dependencies
pip install tensorflow numpy matplotlib jupyter

# Launch the notebook
jupyter notebook digit_recognizer.ipynb
```

No external dataset needed — MNIST loads directly from Keras.

---

## Project structure

```
mnist-digit-recognizer/
│
├── digit_recognizer.ipynb     # Main notebook — run this
├── sample_images/             # Custom digit images for testing
│   ├── digit_7.png
│   └── digit_3.png
├── requirements.txt
└── README.md
```

---

## What I learned

- How CNNs extract spatial features from images using filters
- What Conv2D, MaxPooling, Flatten, and Dropout layers actually do (not just what to call them)
- Why normalization matters before feeding data into a network
- How to read training curves and spot overfitting early
- Interpreting a confusion matrix for multi-class classification

---

## What's next

- Build a live drawing canvas using Gradio or Tkinter where you draw a digit and get a real-time prediction
- Extend to full handwritten word recognition using RNN + CNN
- Try data augmentation (rotations, shifts) to make the model more robust to messy handwriting
- Deploy as a web app using Streamlit

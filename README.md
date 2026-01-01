## Neural Network-Based Accident Type Classification
## Assignment Report

### 1. Introduction
- Objective: Classify accident types using neural networks
- Dataset: 60,004 accident records with 40 features
- Target: Predict 5 accident types (Minor, Property Damage Only, Serious, Fatal, Unknown)

### 2. Data Understanding & EDA (Task 1)
- Initial dataset: 60,004 rows × 54 columns
- Data cleaning: Removed 14 columns with >30% missing values
- Target standardization: Consolidated 6 PDO variations into "Property Damage Only"
- Final dataset: 60,004 rows × 40 columns
- Class distribution: Severe imbalance (Minor: 63.3%, Unknown: 0.04%)

### 3. Data Preparation (Task 2)
- Missing values: Imputed numeric (median) and categorical ("Unknown")
- Encoding: One-hot encoding for 22 categorical features
- Scaling: StandardScaler for 17 numerical features
- Splitting: 70% train, 15% validation, 15% test
- Class weights: Applied to handle imbalance

### 4. Model Design (Task 3)
- Architecture: Feedforward neural network
- Input layer: 308 features (after encoding)
- Hidden layers: 3 layers (256 → 128 → 64 neurons)
- Activation: ReLU for hidden, Softmax for output
- Regularization: Dropout (0.4, 0.3, 0.2) + L2 regularization
- Total parameters: 122,373

### 5. Training Process (Task 4)
- Optimizer: Adam (learning rate: 0.001)
- Loss function: Sparse Categorical Crossentropy
- Batch size: 64
- Epochs: 50 (early stopping at epoch ~25)
- Callbacks: Early stopping, learning rate reduction, model checkpointing
- Training time: ~15 minutes on CPU

### 6. Evaluation (Task 5)
- **Test Accuracy: 99.51%**
- Test Loss: 0.1620
- Weighted F1-Score: 99.53%
- Macro F1-Score: 83.54%
- Confusion matrix: Shows excellent performance except for "Unknown" class

### 7. Interpretation & Recommendations (Task 6)
#### Strengths:
1. Exceptional accuracy (99.51%) on main classes
2. Good generalization (no overfitting)
3. Proper handling of class imbalance
4. Comprehensive preprocessing pipeline

#### Weaknesses:
1. Poor performance on "Unknown" class (insufficient samples)
2. High dimensionality (308 features) after encoding

#### Improvements:
1. Collect more samples for rare classes
2. Consider feature selection to reduce dimensionality
3. Experiment with different architectures (CNN for spatial data)
4. Try ensemble methods or gradient boosting

### 8. Conclusion
- Successfully built neural network for accident classification
- Achieved 99.51% accuracy on test set
- Demonstrated complete ML workflow from data cleaning to evaluation
- Model is ready for deployment with proper monitoring for rare classes

- # neural_network_project.py structure:

# 1. IMPORTS
# 2. DATA LOADING
# 3. DATA CLEANING
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# 5. DATA PREPROCESSING
# 6. NEURAL NETWORK MODEL
# 7. TRAINING
# 8. EVALUATION
# 9. SAVING RESULTS

# **📚 COMPLETE EXPLANATION OF ACTIVATION FUNCTIONS IN YOUR NEURAL NETWORK**

## **🔧 YOUR NEURAL NETWORK'S ACTIVATION FUNCTIONS:**

```python
# Your architecture:
model = keras.Sequential([
    layers.Dense(256, activation='relu'),      # ← ReLU activation
    layers.Dense(128, activation='relu'),      # ← ReLU activation  
    layers.Dense(64, activation='relu'),       # ← ReLU activation
    layers.Dense(5, activation='softmax')      # ← Softmax activation
])
```

## **🧠 1. ReLU (Rectified Linear Unit) - For Hidden Layers**

### **What it is:**
```python
def relu(x):
    return max(0, x)  # Returns x if x > 0, otherwise returns 0
```

### **Why ReLU in YOUR model:**
| Reason | Explanation | Impact on Your Model |
|--------|-------------|---------------------|
| **Non-linearity** | Allows network to learn complex patterns | Can capture complex accident patterns |
| **Computationally efficient** | Simple max(0,x) operation | Fast training on 60K samples |
| **Solves vanishing gradient** | Gradient is either 0 or 1 | Better backpropagation through layers |
| **Sparsity** | Many neurons output 0 | Creates efficient representations |

### **Visual:**
```
Input → ReLU → Output
   -5  →   0
   -1  →   0
    0  →   0
    2  →   2
    5  →   5
```

### **Mathematical:**
```
f(x) = max(0, x)
Derivative: f'(x) = 1 if x > 0, else 0
```

## **🎯 2. Softmax - For Output Layer**

### **What it is:**
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # For numerical stability
    return exp_x / exp_x.sum()
```

### **Why Softmax in YOUR model:**
| Reason | Explanation | Impact on Your Model |
|--------|-------------|---------------------|
| **Multi-class classification** | You have 5 accident types | Perfect for your 5-class problem |
| **Probability distribution** | Outputs sum to 1 | Gives confidence scores for each class |
| **Interpretable outputs** | Each value = probability | Can say "80% chance it's Minor injury" |

### **How it works for YOUR 5 classes:**
```
Raw outputs: [2.1, 1.5, 0.3, -0.5, -1.0]
After Softmax: [0.52, 0.31, 0.10, 0.05, 0.02]  ← Sums to 1.0
Interpretation: 
  - 52% probability → Class 0 (Fatal)
  - 31% probability → Class 1 (Minor)
  - etc.
```

### **Mathematical:**
```
softmax(z_i) = e^{z_i} / Σ_{j=1}^{K} e^{z_j}
where K = 5 (your number of classes)
```

## **📊 COMPARISON TABLE:**

| Function | Used in | Purpose | Output Range | Derivative |
|----------|---------|---------|--------------|------------|
| **ReLU** | Hidden layers | Introduce non-linearity | [0, ∞) | Simple (0 or 1) |
| **Softmax** | Output layer | Multi-class probabilities | [0, 1] (sum=1) | More complex |

## **🤔 WHY NOT OTHER ACTIVATIONS?**

### **For Hidden Layers:**
- ❌ **Sigmoid**: Suffers vanishing gradient, slower (outputs 0-1)
- ❌ **Tanh**: Outputs -1 to 1, still has vanishing gradient issues  
- ✅ **ReLU**: Best for deep networks, fast, prevents vanishing gradient
- ⚠ **Leaky ReLU**: Could use, but ReLU works fine for your case

### **For Output Layer:**
- ❌ **Sigmoid**: Only for binary classification (you have 5 classes)
- ❌ **Linear**: For regression, not classification
- ✅ **Softmax**: Perfect for multi-class classification (your case)

## **🔬 DEEPER EXPLANATION FOR YOUR REPORT:**

### **Why ReLU works for accident data:**
```python
# Accident features like "Number of fatalities" are positive
# ReLU handles this naturally:
fatalities = 3 → ReLU(3) = 3  # Keeps positive values
fatalities = 0 → ReLU(0) = 0  # Handles zero naturally

# Many accident features are counts (≥0)
# ReLU is perfect for this data type
```

### **Why Softmax for accident types:**
```python
# Accident types are MUTUALLY EXCLUSIVE
# An accident can't be both "Fatal" and "Minor"
# Softmax enforces single-class prediction

# Output interpretation:
probabilities = [0.02, 0.85, 0.10, 0.03, 0.00]
# Means: 85% chance it's "Minor" accident
# Clear, interpretable results for decision making
```

## **📈 VISUALIZATION FOR YOUR REPORT:**

```python
# Add this to your report to show activation functions
import numpy as np
import matplotlib.pyplot as plt

# ReLU visualization
x = np.linspace(-10, 10, 100)
y_relu = np.maximum(0, x)

# Softmax visualization (for 3 classes example)
z = np.array([[2.0, 1.0, 0.1],  # Class 1 strongest
              [1.0, 2.0, 0.1],  # Class 2 strongest
              [0.1, 0.2, 3.0]]) # Class 3 strongest
y_softmax = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot ReLU
ax1.plot(x, y_relu, linewidth=3)
ax1.set_title('ReLU Activation Function', fontsize=14)
ax1.set_xlabel('Input')
ax1.set_ylabel('Output')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.2)

# Plot Softmax outputs
classes = ['Fatal', 'Minor', 'PDO']
x_pos = np.arange(len(classes))
width = 0.25

for i in range(3):
    ax2.bar(x_pos + (i-1)*width, y_softmax[i], width, label=f'Sample {i+1}')

ax2.set_title('Softmax Output (Example)', fontsize=14)
ax2.set_xlabel('Accident Type')
ax2.set_ylabel('Probability')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(classes)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
plt.show()
```

## **📝 WHAT TO WRITE IN YOUR REPORT:**

```markdown
### Activation Functions

**Hidden Layers (ReLU):**
- Used Rectified Linear Unit (ReLU) activation
- Formula: f(x) = max(0, x)
- Advantages for our dataset:
  1. Handles positive accident features naturally (counts, ages, etc.)
  2. Prevents vanishing gradient problem in deep network
  3. Computationally efficient for 60,000 samples
  4. Introduces non-linearity to learn complex patterns

**Output Layer (Softmax):**
- Used Softmax activation for multi-class classification
- Formula: σ(z)_i = e^{z_i} / Σ_{j=1}^{K} e^{z_j} where K=5 classes
- Converts raw scores to probability distribution
- Output interpretation: Each value = confidence for accident type
- Ensures mutually exclusive predictions (accident can't be two types)
```

## **✅ SUMMARY FOR YOUR PROJECT:**

**Your activation function choices are PERFECT:**
1. **ReLU** → Best practice for deep neural networks
2. **Softmax** → Standard for multi-class classification  
3. **Combination** → Industry standard for your problem type

**No changes needed!** Your design follows best practices for:
- Binary/multi-class classification
- Deep neural networks
- Real-world tabular data
- Academic/professional standards

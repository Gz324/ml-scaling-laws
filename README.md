# Scaling Laws and Generalization in Deep Learning

This project explores how model capacity and dataset size influence generalization performance in deep neural networks.

---

## Motivation

Modern deep learning systems often improve with more data and larger models, but the relationship between these factors and generalization is not always straightforward.

This project aims to empirically study:
- how increasing dataset size affects performance  
- how model capacity impacts overfitting  
- how the generalization gap evolves  

---

## Methodology

- Trained convolutional neural networks with varying widths  
- Used subsets of CIFAR-10 with sizes:
  - 1K, 5K, 10K samples  
- Measured:
  - training accuracy  
  - test accuracy  
- Analyzed generalization gap across configurations  

---

## Results

- Larger datasets reduce overfitting and improve generalization  
- Smaller datasets lead to higher training accuracy but poorer test performance  
- Model capacity interacts with data size in determining performance  

### Visualization

![Scaling Results](scaling_results.png)

---

## Key Insight

Generalization depends not only on model size, but also on the amount of data available.  
Balancing these factors is critical for building effective learning systems.

---

## Run

```bash
pip install -r requirements.txt
python3 scaling_laws.py

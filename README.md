# CIL ETH Project🧪


This repo is for all our experiments related to the CIL project at ETH. It's structured to make it easy to plug in new models, configs, datasets, etc.

---

## 🗂 Repo Structure

### configs/ ⚙️  
Contains training setup:
- Hyperparameters (learning rate, batch size, etc.)
- Augmentations
- Model/optimizer/loss initialization

### datasets/ 📚  
PyTorch Dataset class for defining (train_x, label_y) pairs.  
Originally taken from the baseline – works fine as-is.

### models/ 🧠  
Model architectures:
- Both custom and baseline versions
- Naming convention:
  - <model_name>.py → main architecture
  - <model_name>_utils.py → extra blocks if needed

### utils/ 🛠  
Currently has train_utils.py with train/validation/test loops used in notebooks. You may also add here any additional supplementary functions.

### notebooks/ 📓  
Where experiments are run:
- Import configs, split data, train/evaluate models
- Includes a notebook to download data (requires Kaggle API key)

---

## ⚠️ Before You Run Anything

- Make sure to set the correct data path and preferred GPU in configs or notebooks.
- For the data download notebook, your Kaggle API key is needed (get it from your [Kaggle account](https://www.kaggle.com/settings)).

---

## 💡 Potential Improvements

Some areas we could explore next:

- Augmentations: Right now it’s just the baseline setup. Could try things like flipping, perspective shifts, etc. Just keep in mind the target/label needs to be transformed too.
- Model architectures: Only one custom model so far. Lots of other designs to test out or build on top of.
- Losses/optimizers: Using AdamW + MSE for now. Since our metric is scale-invariant RMSE, maybe MSE isn’t ideal. Might be worth trying alternatives or optimizing the metric directly.

---

## ✅ To-Do / Wishlist
- [ ] Add more advanced augmentations
- [ ] Try different model variants
- [ ] Run ablation studies on loss functions, optimizers, and hyperparameters

---

Hope that helps!
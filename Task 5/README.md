# 🏠 Multimodal Housing Price Prediction

## 📖 Overview
This project predicts **housing prices** using **two types of data**:
- **Images** of houses (processed using a CNN)
- **Tabular data** such as square footage, bedrooms, bathrooms, year built, and neighborhood

The model combines **image features** and **tabular features** into a **multimodal neural network** for regression.

---

## 🎯 Objectives
- Use **CNNs** to extract features from images
- Combine with **structured/tabular data**
- Train a **multimodal regression model**
- Evaluate using **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**

---

## 📂 Dataset
The dataset consists of:
- **`housing_data.csv`** → metadata (price, sqft, bedrooms, bathrooms, year built, neighborhood, image path)
- **`images/` folder** → house images

Example CSV format:

```csv
image_path,price,sqft,bedrooms,bathrooms,year_built,neighborhood
images/house1.jpg,250000,1200,3,2,2005,A
images/house2.jpg,320000,1500,4,3,2010,B
images/house3.jpg,180000,900,2,1,1995,A
images/house4.jpg,450000,2000,5,3,2018,C
images/house5.jpg,380000,1700,4,2,2012,B
images/house6.jpg,220000,1100,3,1,2000,A
images/house7.jpg,500000,2200,5,4,2020,C
images/house8.jpg,275000,1300,3,2,2008,B

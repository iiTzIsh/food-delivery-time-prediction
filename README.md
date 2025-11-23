# 🍽️ Food Delivery Time Prediction

A machine learning-powered web application that predicts food delivery time based on multiple factors including distance, traffic conditions, weather, and delivery person characteristics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Data Processing](#data-processing)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a **Random Forest Regression** model to predict food delivery times with high accuracy. The application features an interactive web interface built with Streamlit, allowing users to input various delivery parameters and receive instant time predictions.

### Key Highlights:
- ✅ **Accurate Predictions**: Random Forest model trained on real-world delivery data
- ✅ **Interactive UI**: User-friendly Streamlit interface
- ✅ **Real-time Inference**: Instant predictions based on current conditions
- ✅ **Multiple Features**: Considers 11+ factors affecting delivery time
- ✅ **Robust Pipeline**: Complete preprocessing and prediction pipeline

## ✨ Features

### Prediction Factors:
1. **📍 Delivery Information**
   - Distance (km)
   - Rider age
   - Rider ratings
   - Order-to-pickup time

2. **🌍 Environmental Conditions**
   - Traffic density (low, medium, jam)
   - Weather conditions (sunny, stormy, fog, windy, sandstorms)
   - City type (urban, semi-urban, metropolitan)
   - Festival periods

3. **📦 Order Details**
   - Order type (meal, snack, drinks)
   - Vehicle type (motorcycle, scooter)
   - Multiple deliveries count

## 🖼️ Demo

The application consists of two main pages:

### Home Page
Welcome screen with navigation to the predictor

### Predictor Page
Interactive form with real-time predictions:
- **Input Fields**: Organized in 3-column layout for easy data entry
- **Instant Predictions**: Get delivery time estimates in minutes
- **Smart Feedback**: Color-coded feedback based on predicted time

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Guide

1. **Clone the repository**
```bash
git clone https://github.com/iiTzIsh/food-delivery-time-prediction.git
cd food-delivery-time-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify model file**
Ensure `rf_pipeline.pkl` exists in the root directory. If not, run the training script:
```bash
python new_food_delivery.py
```

## 💻 Usage

### Running the Application

1. **Start the Streamlit app**
```bash
streamlit run app.py
```

2. **Open your browser**
The application will automatically open at `http://localhost:8501`

3. **Navigate to Predictor**
Click the "Go to Predictor" button on the home page

4. **Enter delivery details**
- Fill in all required fields
- Click "Predict Delivery Time"
- View the estimated delivery time

### Example Input
```
Distance: 5.0 km
Rider Age: 28
Rider Rating: 4.5
Pickup Time: 10 minutes
Traffic: medium
Weather: sunny
City: urban
Festival: no
Order Type: meal
Vehicle: motorcycle
Multiple Deliveries: 1
```

### Expected Output
```
🕒 Estimated delivery time: 28.5 minutes
🚚 Normal delivery time
```

## 📁 Project Structure

```
food-delivery-time-prediction/
│
├── app.py                          # Main Streamlit application (home page)
├── pages/
│   └── predictor.py                # Prediction page with model inference
│
├── new_food_delivery.py            # Complete ML pipeline & training script
├── New_food_delivery.ipynb         # Jupyter notebook with EDA & experiments
│
├── rf_pipeline.pkl                 # Trained Random Forest model pipeline
│
├── train.csv                       # Original training dataset
├── cleaned_food_delivery.csv       # Cleaned dataset
├── food_delivery_preprocessed.csv  # Final preprocessed data
│
├── images/
│   └── food_delivery.jpg           # UI image
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🤖 Model Details

### Algorithm: Random Forest Regressor

**Hyperparameters:**
```python
n_estimators=100          # Number of trees
max_depth=15              # Maximum tree depth
min_samples_split=10      # Minimum samples to split node
min_samples_leaf=5        # Minimum samples in leaf node
random_state=42           # Reproducibility
n_jobs=-1                 # Parallel processing
```

### Performance Metrics

| Metric | Score |
|--------|-------|
| **R² Score** | High accuracy on test set |
| **MAE** | Low mean absolute error |
| **RMSE** | Robust root mean squared error |

### Feature Importance
Top features influencing predictions:
1. **Distance_km** - Most significant factor
2. **Order_to_pickup_min** - Preparation time impact
3. **Traffic density** - Road conditions
4. **Weather conditions** - Environmental factors
5. **Delivery person ratings** - Experience indicator

### Model Pipeline
```
Input Data
    ↓
Column Transformer
    ├── Numeric Features → StandardScaler
    └── Categorical Features → OneHotEncoder
    ↓
Random Forest Regressor
    ↓
Predicted Delivery Time (minutes)
```

## 🔧 Data Processing

### 1. Data Cleaning
- Removed missing values and invalid entries
- Stripped whitespace and standardized text
- Converted data types appropriately
- Filtered unrealistic values:
  - Age: 18-65 years
  - Ratings: 1-5 scale
  - Distance: ≤50 km
  - Delivery time: ≤120 minutes

### 2. Feature Engineering

**Calculated Features:**
- **Distance_km**: Haversine formula for geodesic distance
  ```python
  haversine_distance(restaurant_lat, restaurant_lon, 
                     delivery_lat, delivery_lon)
  ```

- **Order_to_pickup_min**: Time between order and pickup
  ```python
  (Time_Order_picked - Time_Orderd) in minutes
  ```

- **Temporal Features**:
  - Day, Month, Weekday
  - Weekend indicator
  - Hour of day
  - Peak hours (11-14, 18-21)

**Text Normalization:**
- Lowercase conversion
- Prefix removal ("conditions " from weather)
- Category standardization

### 3. Data Transformation

**Numeric Features:**
- Delivery_person_Age
- Delivery_person_Ratings
- Distance_km
- Order_to_pickup_min
- multiple_deliveries

**Categorical Features:**
- Weatherconditions
- Road_traffic_density
- Type_of_order
- Type_of_vehicle
- Festival
- City

### 4. Train-Test Split
- **Training Set**: 80%
- **Test Set**: 20%
- **Random State**: 42 (for reproducibility)

## 🛠️ Technologies Used

### Core Framework
- **Python 3.8+**: Primary programming language
- **Streamlit 1.50.0**: Web application framework

### Machine Learning
- **scikit-learn 1.6.1**: ML algorithms and preprocessing
- **NumPy 2.2.2**: Numerical computations
- **Pandas 2.3.3**: Data manipulation

### Model Persistence
- **Joblib 1.5.2**: Model serialization

### Development Tools
- **Jupyter Notebook**: Exploratory data analysis
- **Matplotlib**: Data visualization (during training)

## 📊 Dataset Information

### Original Dataset Features
- **ID**: Unique identifier
- **Delivery_person_ID**: Rider identifier
- **Delivery_person_Age**: Age of delivery person
- **Delivery_person_Ratings**: Customer ratings (1-5)
- **Restaurant_latitude/longitude**: Restaurant coordinates
- **Delivery_location_latitude/longitude**: Delivery coordinates
- **Order_Date**: Date of order
- **Time_Orderd**: Order placement time
- **Time_Order_picked**: Pickup time
- **Weatherconditions**: Weather at delivery time
- **Road_traffic_density**: Traffic conditions
- **Type_of_order**: Food category
- **Type_of_vehicle**: Delivery vehicle
- **multiple_deliveries**: Number of orders in batch
- **Festival**: Festival period indicator
- **City**: City type
- **Time_taken(min)**: **Target variable** - Actual delivery time

## 🔍 Model Training Process

1. **Load and clean data** from `train.csv`
2. **Engineer features** (distance, time differences, temporal features)
3. **Remove outliers** and validate data quality
4. **Create preprocessing pipeline** (scaling + encoding)
5. **Train Random Forest model** with optimized hyperparameters
6. **Evaluate performance** on test set
7. **Save pipeline** as `rf_pipeline.pkl`
8. **Export preprocessed data** for future use

## 🚧 Known Issues & Solutions

### scikit-learn Version Compatibility
The predictor includes automatic compatibility fixes for different sklearn versions:
```python
# Handles version mismatches gracefully
# Falls back from joblib to pickle if needed
# Provides clear error messages with fix instructions
```

### Missing Model File
If `rf_pipeline.pkl` is missing:
```bash
python new_food_delivery.py
# This will retrain and save the model
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contributions
- 🎨 Improve UI/UX design
- 📈 Add more visualizations
- 🧪 Implement additional ML models
- 📱 Mobile responsiveness
- 🌐 Multi-language support
- 🔔 Add notification features

## 📝 Future Enhancements

- [ ] Real-time tracking integration
- [ ] Historical data visualization dashboard
- [ ] API endpoint for external integrations
- [ ] Mobile application
- [ ] A/B testing for model improvements
- [ ] Deployment to cloud platforms (AWS, GCP, Azure)
- [ ] Docker containerization
- [ ] CI/CD pipeline setup

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ishara Madusanka**
- GitHub: [@iiTzIsh](https://github.com/iiTzIsh)
- Project: [Food Delivery Time Prediction](https://github.com/iiTzIsh/food-delivery-time-prediction)

## 🙏 Acknowledgments

- Dataset source: Food delivery platform data
- Streamlit community for excellent documentation
- scikit-learn for powerful ML tools
- Open-source community for inspiration

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/iiTzIsh/food-delivery-time-prediction/issues) page
2. Create a new issue with detailed description
3. Contact the maintainer

---

⭐ **Star this repository** if you found it helpful!

Made with ❤️ by Ishara Madusanka

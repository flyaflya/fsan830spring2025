# Horse Race Prediction Model

## Project Objective
This project aims to predict horse race outcomes using a Bayesian Additive Regression Trees (BART) model. The model leverages historical race data and various features to predict the likelihood of each horse's finishing position in a race.

## Model Architecture

### 1. Data Processing
- **Feature Integration**: Combines race-specific features, recent performance metrics, and jockey/trainer statistics
- **Feature Engineering**: Creates relative performance measures and derived features
- **Data Validation**: Ensures data consistency and handles missing values

### 2. Feature Selection
The model uses features that are available in both training and prediction datasets:

#### Race-specific Features
- Surface type (dirt, turf, etc.)
- Distance
- Purse amount
- Program number

#### Recent Performance Features
- Recent finish position
- Recent surface type
- Recent distance
- Recent purse amount
- Recent starting position

#### Jockey/Trainer Features
- Jockey name and statistics (wins, places, shows)
- Trainer name and statistics (wins, places, shows)
- Combined trainer-jockey performance metrics

### 3. BART Model
- **Implementation**: Uses PyMC's BART implementation
- **Advantages**: 
  - Handles non-linear relationships
  - Provides uncertainty estimates
  - Robust to outliers
  - Can capture complex interactions between features

### 4. Prediction Proxy
- **Target Variable**: Finish position (1st, 2nd, 3rd, etc.)
- **Output**: Probability distribution of finishing positions for each horse
- **Ranking**: Horses are ranked based on their predicted probabilities of winning

## Model Features

### 1. Race-specific Features
- Surface type (dirt, turf, etc.)
- Distance
- Purse amount
- Program number

### 2. Recent Performance Features
- Recent finish position
- Recent surface type
- Recent distance
- Recent purse amount
- Recent starting position

### 3. Jockey/Trainer Features
- Jockey name and statistics
- Trainer name and statistics
- Combined trainer-jockey performance metrics

### 4. Relative Performance Measures
- Relative finish position
- Relative purse amount
- Relative starting position

## Model Training

### 1. Data Preparation
- Load and preprocess historical race data
- Calculate derived features
- Handle missing values
- Create relative performance metrics

### 2. Feature Engineering
- Calculate jockey and trainer statistics
- Create relative performance measures
- Handle categorical variables through one-hot encoding

### 3. Model Training
- Train BART model on historical data
- Use cross-validation to tune hyperparameters
- Monitor model performance metrics

## Model Evaluation

### 1. Performance Metrics
- Mean Absolute Error (MAE) of predicted vs. actual finish positions
- Top-3 prediction accuracy
- Win prediction accuracy

### 2. Validation Strategy
- Time-based cross-validation
- Out-of-sample testing
- Performance on different race types and conditions

## Model Deployment

### 1. Prediction Pipeline
- Load and preprocess new race data
- Apply feature engineering
- Generate predictions using trained model
- Output probability distributions for each horse

### 2. Output Format
- Race ID
- Horse ID
- Predicted finish position
- Probability of winning
- Confidence intervals

## Usage

### Training the Model
```bash
python src/train.py
```

### Making Predictions
```bash
python src/predict.py
```

## Dependencies
- Python 3.8+
- PyMC
- NumPy
- Pandas
- Xarray
- scikit-learn

## Future Improvements
1. Incorporate more advanced feature engineering
2. Implement ensemble methods
3. Add real-time odds integration
4. Develop a web interface for predictions
5. Add more sophisticated validation strategies
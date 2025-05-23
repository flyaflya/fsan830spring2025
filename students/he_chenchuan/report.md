# Horse Racing Prediction Project: A Journey of Learning

One note before everything starts: the data files and generated models are deleted in my source code for better repo management.

In this project, I tried on two attempts to build a horse racing prediction model, each guided by different approaches to AI-assisted development. 

After persistent failures, I finally investigated the fundamental issue: the systematic differences between training and prediction datasets. Upon detailed code inspection, I discovered that the core problem wasn't with the model or implementation, but rather with the data itself - almost no features were shared between the two data sources.

Below I will describe my coding initiative and phylosophy, and the stage that I have reached, after a short description of the data failure issue. And finally with a reflection with vibe coding issue of AI, the one guidance that should always focus first.

# The Data Mismatch Challenge

To investigate the feature compatibility between training and prediction datasets, I followed the same code paths as the reference implementation:
- Training data processing: `students/fleischhacker_adam2/src/parse_race_results.py`
- Prediction data processing: `students/fleischhacker_adam2/src/1CleanPredictionData.py`

# The data unmatched issue
To checked the features that might be fitted for both training and prediction dataset, I follow the same code `students/fleischhacker_adam2/src/parse_race_results.py` to get the prossed training data from .xml files. And following the same code `students/fleischhacker_adam2/src/1CleanPredictionData.py` to get the prediction data.

After that, I read the processed prediction data, removing columns with all NaN values, and columns without a real column name (thouse start with `col_`) after preprocessing. 50 features left, then I check what are the columns for current race and what are the columns for past race, and only keep the columns with current race as the most vaiable produce features that might be at least used for checking the data matching. 

So that I ask GPT-4o in ChatGPT, 
```text
below are the columns from two different data sources about horse racing, (so that for the same feature, the names might be different.). I want you know to extract all the paired featurs as a json object map. 
```

Then ChatGPT returned to me the map:
```json
{
  "track_code": "race", 
  "race_number": "race", 
  "program_number": "program_number",
  "odds": "odds",
  "horse_name": "horse",
  "jockey": "jockey",
  "trainer_name": "trainer",
  "purse": "purse",
  "surface_code": "surface",
  "distance": "distance_f"
}
```

The mapping revealed a critical issue: key features like horse names and trainers had no matching entries between the datasets. Even more concerning, the jockey column was completely absent from the training dataset. This fundamental mismatch meant we couldn't use the most important features that indicate a horse's racing ability, making reliable predictions impossible.

# Failed Trail 1

In the first trail, I attempted to build a BART (Bayesian Additive Regression Trees) model for horse race prediction. The approach involved:

1. **Data Processing**
   - Parsed XML racing data from past performance files
   - Extracted comprehensive features including:
     - Current race details (track, surface, distance, purse)
     - Horse information (name, jockey, trainer)
     - Past performance metrics (finish positions, lengths back, speed figures)
   - Standardized various data formats (e.g., odds, distances)

2. **Model Architecture**
   - Implemented a BART model using PyMC and PyMC-BART
   - Designed a Bayesian framework with:
     - Normal likelihood for the target variable
     - Half-normal prior for the error term
     - BART prior for the mean function
   - Included MCMC sampling with configurable parameters (draws, tuning, chains)

3. **Training Process**
   - Split data into training and validation sets
   - Implemented feature scaling and encoding
   - Conducted MCMC sampling
   - Evaluated using multiple metrics (MSE, R², Spearman correlation)

4. **Prediction Variables**:
   - Primary target: `finish_position` (horse's finishing position in the race)
   - Input features included:
     - Race features: `surface`, `distance_f`, `purse`
     - Horse features: `horse_name`, `program_number`
     - Past performance: `recent_finish_pos`, `recent_lengths_back_finish`, `recent_speed_figs`
     - Participant features: `jockey`, `trainer`

5. **Results and Issues**:
   - The model failed to perform well due to the fundamental data mismatch issue
   - Even with sophisticated feature engineering and a powerful BART model, the lack of matching features between training and prediction datasets made it impossible to make reliable predictions
   - This led to the realization that we needed to first address the data compatibility issue before attempting any modeling

# Failed Trail 2

In the second trail, I took a more structured and focused approach, due to the repeated data matching failure learning from the issues encountered in the first trail. The key differences and improvements were:

1. **Data Processing Improvements**:
   - Created a dedicated `HorseRaceDataProcessor` class for more organized data handling
   - Focused on a smaller, more relevant set of features:
     - Core race features (surface, distance, purse)
     - Recent performance metrics (finish position, surface, post position)
     - Jockey and trainer information
   - Added derived features:
     - Field size calculations
     - Relative performance metrics (compared to field averages)
     - Jockey and trainer statistics
   - Better handling of missing values with appropriate imputation strategies

2. Model Architecture Refinements:
   - Simplified BART model implementation with clearer parameter management
   - Added noise to predictions to break ties between horses
   - Included uncertainty estimation in predictions
   - More focused hyperparameter tuning:
     - Increased number of trees (200 vs previous implementation)
     - More chains (4 vs 2) for better posterior sampling
     - Configurable random seed for reproducibility

3. Training Process Enhancements:
   - More robust feature preparation pipeline
   - Better handling of training/testing data splits
   - Added validation for data structure consistency
   - Improved race-level prediction aggregation

4. Prediction Variables:
   - Primary target: `finish_position` (horse's finishing position in the race)
   - Input features were more focused and included:
     - Race features: `surface_code`, `distance`, `purse`
     - Recent performance: `recentFinishPosition1`, `recentSurfaceCode1`, `recentPostPosition1`
     - Participant features: `jockey_name`, `trainer_name`
     - Derived features:
       - `field_size`: Number of horses in the race
       - `relative_finish`: Finish position relative to field average
       - `relative_purse`: Purse relative to field average
       - `relative_start`: Starting position relative to field average
       - `jockey_starts`, `jockey_avg_finish`: Jockey performance statistics
       - `trainer_starts`, `trainer_avg_finish`: Trainer performance statistics

5. Results and Lessons:
   - Despite the improvements in data processing and model architecture, the fundamental data mismatch issue persisted
   - The more focused feature set and better data handling didn't overcome the lack of matching features between training and prediction datasets
   - This reinforced the importance of data compatibility as the primary concern before any modeling attempts

The second trail demonstrated that even with better code organization, more sophisticated feature engineering, and improved model implementation. However, due to the data mismatch issue, although a model can be trained, even used to predict with the prediction data. However, the result is unbelievably strange. 

# Final Remarks and Reflections

The most important lesson I learned from this project is about the critical balance between AI assisted vibe coding and human understanding. While AI tools can be powerful aids in development, they should not replace fundamental human comprehension of the problem space.

Throughout this project, I discovered that blindly handing off tasks to AI without first understanding the underlying data and requirements leads to wasted effort. Just as I instruct AI assistants to "not guess the code but search the codebase for actual logic. If you still have concerns, ask me for clarification". I found myself making the same mistake - acting like an AI that hallucinates about data structures without verifying the reality.

So, really pay attention to the problem, understand the problem, the data structure, do some preliminary data analysis, and then, start your modeling process. Otherwise, it can be a totally waste of effort. One step further, I think this further reveal the dark side of AI-assistants. As most students can use AI to finish the homeworks, they can solve the homework problems without actually understand the real situation. Even when they are wrong, they may simply blame that it is the fault of AI being not compatiable enough. So, I myself, in this project, shows as an example of trying to solve a problem with AI assistants without actually understanding the problem comprehensively, which is what I think a really dangerous zone, that may harm the whole education system. While it also point out one direction for future assessment of academic excellence, how to define the problems that AI cannot solve without proper human guidances.
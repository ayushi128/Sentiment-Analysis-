# Sentiment-Analysis-
This code uses ML model to analyze sentiments in a tweet or comment.
# Sentiment Analysis Using Logistic Regression and TF-IDF

## Description
This project implements a sentiment analysis pipeline that classifies tweets into positive, negative, or neutral sentiments. It uses Python and scikit-learn with TF-IDF vectorization and a Logistic Regression model for classification.

## Dataset
The dataset used is the Twitter Sentiment Analysis Dataset from Kaggle. It contains labeled tweets used for training and validating the model.

## Technologies Used
- Python 3.x
- pandas
- NLTK
- scikit-learn
- matplotlib
- seaborn

## Installation
To install the required libraries, run:

## Usage
1. Clone or download the repository.
2. Place the dataset CSV files (`twitter_training.csv`, `twitter_validation.csv`) in the `data/` folder.
3. Run the Jupyter notebook or Python script `sentiment_analysis_demo.ipynb` / `sentiment_analysis_demo.py`.
4. The notebook/script preprocesses the data, trains the model, and outputs accuracy and confusion matrix.

## Project Structure
- `data/` - Folder containing dataset CSV files.
- `sentiment_analysis_demo.ipynb` - Jupyter notebook with code and explanations.
- `README.md` - This file.
- `demo_video.mp4` - Demo video of the project (optional).

## Results
The model achieves approximately 91.1% accuracy on validation data. The confusion matrix provides a detailed summary of correct and incorrect predictions per sentiment class.

## Future Work
- Improve preprocessing using advanced NLTK features.
- Test other classifiers like Naive Bayes and SVM.
- Use larger or different datasets for better performance.

## Contact
Ayushi Bansal  
Email: ayushibansal128@gmail.com  
GitHub: https://github.com/ayushi128 

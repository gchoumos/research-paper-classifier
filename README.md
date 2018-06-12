# Research Paper Classifier - Kaggle Competition

Link to the competition: https://www.kaggle.com/c/m35216f-article-classification-challenge

This repository holds my implementation of a Machine Learning project in the context of the **Data Science Challenge** course. This course is part of the Master's Degree in Data Science of the Athens University of Economics and Business.

The main idea is that we are given textual and graph-theoretical information regarding scientific articles

- ID
- Abstract
- Title
- Authors
- Publication year
- Citations

and we are asked to predict the category of each. There are 28 different categories that are based on the journal in which they were published. 

The project is not yet ready, although the current results are promising. The next steps will be

- ~~Code-refactoring of the `data_preprocessor.py` - Currently is a script but it should become a class.~~
- ~~Integrate data preparation with the main program (`text_classifier.py`)~~
- ~~Ask the user if he wants to recreate the datasets.~~
- ~~Create a settings file and start adding there all the configuration.~~
- Code-refactor the `text_classifier.py`.
- Validation on Kaggle was supposed to be temporary. Hold out some of the data.
- Generate some learning curves.
- Try to apply some topic modelling and then check if adding the most probable topic as a feature improves accuracy.
- If the previous works (or even if it doesn't) try and use LDAvis as it offers some cool visualisations for topics.
- Apply feature selection and make sure that anything that exists in the Feature Union is indeed informative.
- Add visualisations for the various features, probably by importing the data to Tableau.

This README file will be soon updated with a detailed walk through the code and a description of the plan of attack to the problem.

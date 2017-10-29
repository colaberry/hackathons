# University of Texas, Dallas - Hackathon Oct 2017

# Feedback link : https://goo.gl/forms/iSRbrK8cOUou8YaG3

Hackathon URL: http://hackathon.refactored.ai
Language: Python 3

Here are instructions for the hackathon as well as scoring guidelines. There are 3 problems that you can pick and choose from based on their level of complexity. Note that easier ones too have bonus points for additional work. 

## Hack Away
If you have already used jupyter notebooks, then it is easy to get started:

1. Click on http://hackathon.refactored.ai and access your notebook environment.
2. Click on new and select python 3 notebook.
3. Edit the sample lesson
4. Validate the lesson by clicking on 'Validate'
5. Test the lesson by clicking on 'Test'
6. Continue to add new lessons by clicking on 'MakeLesson'

* For detailed instructions specific to using refactored look at the word document: Hackathon_at_UTD_Fall17.docx

Problem 1: EDA and Prediction of Graduate Employment Rate (GER) on Grad Students data

Data Link: https://github.com/colaberry/data/blob/master/Grad_Students/grad-students.csv

History of Data
The dataset is taken from American Community Surveys (ACS) and pertains to graduate employment related data within the years 2010 to 2012.
The dataset is part of a master dataset found at the following link: https://factfinder.census.gov/faces/nav/jsf/pages/searchresults.xhtml?refresh=t
Loading the dataset
Pandas library in Python is one of the most widely used libraries for data analysis, due to the functions and features it provides. The dataframe is an object which allows storage of a table. The given dataset can be loaded into a dataframe for analysis. The read_csv function of Pandas library can read contents of a csv file into a dataframe.

* Submit the EDA as a course: 50 points
* Perform prediction of GER (using linear regression only) on the dataset: 40 points
* Submit to GitHub: 10 points
* Level : EDA-Easy, Prediction-Difficult

Naming Convention:
* Name your Python file as "Team Name"_grad_students

Problem 2: EDA for the Women in STEM dataset

Data: https://github.com/fivethirtyeight/data/blob/master/college-majors/women-stem.csv
## History

Data is originally from American Community Survey 2010-2012 Public Use Microdata Series.
The dataset consists of 75 STEM major studies divided into 5 categories. It also gives the number of men and women in each major along with the median salary and the proportion of women(ShareWomen) in each STEM major.

## Exercise

* Submit the work as a course: 50 points
* Export it to GitHub/Submit as lab: 10 points
* Perform data cleaning and explanatory data analysis on this dataset. 10 points
* Provide visualizations and derive conclusions. 20 points
* Split the dataset into training and testing at 80:20 ratio. Use random seed = 12345. 10 Points
* Fit a linear regression model to predict the ShareWomen. Use model name stem_model. Bonus +10 points
* Predict ShareWomen using the model and calculate mean squared error for training and testing datasets. +10 points
Level: Medium

## Use the following variable names

* Model name stem_model.
* MSE_Train, MSE_Test for the mean squared error and Pred_Test for predictions on the test dataset.

Naming Convention:
* Name your Python file as "Team Name"_stem_women
  
# Problem 3: Somerville Happiness Survey

Data : https://data.somervillema.gov/api/views/yevj-2b33/rows.csv
## About the Dataset
The City of Somerville, MA sends out a happiness survey to a random sample of Somerville residents asking them to rate their personal happiness and their satisfaction with City services. This combined dataset includes the survey responses for years 2011, 2013, and 2015
The data set consists of survey responses of Somerville residents for the years 2011, 2013, and 2015.
Some of the questions asked in the survey include:
* What.is.your.gender._2011
* Age
* Marital.status._2011
* How.long.have.you.lived.here.
* What.is.your.annual.household.income.
* How.happy.do.you.feel.right.now.
* How.satisfied.are.you.with.your.life.in.general.
* How.satisfied.are.you.with.your.neighborhood.

## Exercise
* Export as a course: 50 points
* Perform the data cleaning of the somerville happiness survey dataset: 10 points
* Analyze the survey response and demographic data to understand the data : 10 points
* Provide the visualizations for the data to aid in the analysis: 10 points
* Fit the model using logistic regression : 20 points
* Export to GitHub: 10 points

Bonus: 
* Calculate the Mean Square Error (MSE) for the test predictions : 10 points
* Use the following variable names 
* Ensure the below variables are available in the notebook for evaluation:
* happiness_model - representing the logistic regression model to fit the data : 10 points
* MSE_Test - variable containing the mean squared error : 5 points
* Y_HAT - containing the predictions on the test dataset. : 5 points

Level : Difficult.

Naming Convention:

* Name your Python file as "Team Name"_happy_somer


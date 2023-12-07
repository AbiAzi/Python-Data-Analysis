'''
1. Problem Statement
    This project aims to understand how student performance (test scores) is influenced by various factors,
     including Gender,
    Ethnicity, Parental Level of Education, Lunch, and Test Preparation Course.
2. Data Collection
Dataset Source: Kaggle Dataset
The dataset consists of 8 columns and 1000 rows.
    2.1 Dataset Information
    Gender: Sex of students → (Male/Female)
    Race/Ethnicity: Ethnicity of students → (Group A, B, C, D, E)
    Parental Level of Education: Parents' final education → (Bachelor's degree, Some college, Master's degree,
    Associate's degree, High school)
    Lunch: Having lunch before test → (Standard or Free/reduced)
    Test Preparation Course: Complete or not complete before test
        .Math Score €Calculator;
        .Reading Score
        .Writing Score
2.2 Import Data and Required Packages
Importing Pandas, Numpy, Matplotlib, Seaborn, and Warnings Library.

''';
import inline
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Get basic statistics of the dataset
print("===========================================================================================================")
print()
print("1, Get basic statistics of the dataset\n")

print()
df = pd.read_csv("StudentsPerformance.csv")
# Display the first and last 5 rows of the DataFrame
print("The first 5 rows of DataFrame: \n",df.head(5))
print("The last 5 rows of DataFrame: \n",df.tail(5))
print("Basic statistics of the dataset: \n",df.describe())
print("==>> Insights From above description of numerical data, all means are very close to each other ")
print("between 66 and 68.05 ")

print("All standard deviations are also close - between 14.6 and 15.19;While there is a minimum score 0 ")
print("for math, for writing minimum is much higher = 10 and for reading might higher = 17")

#Shape of the Dataset

print()
print("Shape of theis dataset is: ")
print(df.shape)
#count number of female and male (gender values) in the data
print("Number of female and male students: \n",df.gender.value_counts())
#List out students who get 100% performance.
print("Students who get 100% performance: \n",df[(df["math score"] == 100) & (df["reading score"] == 100 )
& (df["writing score"] == 100 )])
print("=============================================================================================================")
print()
print("2, Data Checks to Perform")
print()
'''Data Checks to Perform
    .Check Missing Values
    .Check Duplicates
    .Check Data Type
    .Check the Number of Unique Values of Each Column
    .Check Statistics of Data Set
    .Check Various Categories Present in Different Categorical Columns'''
# Check for any missing values
print("Null values found: \n",df.isnull().sum())
print(df.info())
# Check Duplicate
print("No. of Duplicated data: ",df.duplicated().sum());
# Checking the Number of Unique Values of Each Column
print("Number of Unique Values of Each Column: \n",df.nunique())
print("==========================================================================================================")
print()
print("3, Exploring Data")
print()
#Check categories in each columns
print("Categories in 'gender' variable:   ", end=" ")
print(df['gender'].unique())

print("Categories in 'race/ethnicity' variable:  ", end=" ")
print(df['race/ethnicity'].unique())

print("Categories in 'parental level of education' variable:", end=" ")
print(df['parental level of education'].unique())

print("Categories in 'lunch' variable:     ", end=" ")
print(df['lunch'].unique())

print("Categories in 'test preparation course' variable:     ", end=" ")
print(df['test preparation course'].unique())

# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
#Adding Columns for Total Score and Average
df['total score'] = df['math score'] + df['reading score'] + df['writing score']
df['average'] = df['total score'] / 3
print(df.head())

# Count the number of students with full marks (100) in each subject
reading_full = df[df['reading score'] == 100]['average'].count()
writing_full = df[df['writing score'] == 100]['average'].count()
math_full = df[df['math score'] == 100]['average'].count()

# Print the count of students with full marks in each subject
print(f'Number of students with full marks in Maths: {math_full}')
print(f'Number of students with full marks in Writing: {writing_full}')
print(f'Number of students with full marks in Reading: {reading_full}')

# Calculate the percentage of students with full marks in each subject
total_students = len(df)
percentage_math_full = (math_full / total_students) * 100
percentage_writing_full = (writing_full / total_students) * 100
percentage_reading_full = (reading_full / total_students) * 100

print()
# Print the percentage of students with full marks in each subject
print(f'Percentage of students with full marks in Maths: {percentage_math_full:.2f}%')
print(f'Percentage of students with full marks in Writing: {percentage_writing_full:.2f}%')
print(f'Percentage of students with full marks in Reading: {percentage_reading_full:.2f}%')

print()
# Count the number of students with less than or equal to 20 marks in each subject
reading_less_20 = df[df['reading score'] <= 20]['average'].count()
writing_less_20 = df[df['writing score'] <= 20]['average'].count()
math_less_20 = df[df['math score'] <= 20]['average'].count()

# Print the count of students with less than or equal to 20 marks in each subject
print(f'Number of students with less than or equal to 20 marks in Maths: {math_less_20}')
print(f'Number of students with less than or equal to 20 marks in Writing: {writing_less_20}')
print(f'Number of students with less than or equal to 20 marks in Reading: {reading_less_20}')

# Calculate the percentage of students with less than or equal to 20 marks in each subject
total_students = len(df)
percentage_math_less_20 = (math_less_20 / total_students) * 100
percentage_writing_less_20 = (writing_less_20 / total_students) * 100
percentage_reading_less_20 = (reading_less_20 / total_students) * 100
print()
# Print the percentage of students with less than or equal to 20 marks in each subject
print(f'Percentage of students with less than or equal to 20 marks in Maths: {percentage_math_less_20:.2f}%')
print(f'Percentage of students with less than or equal to 20 marks in Writing: {percentage_writing_less_20:.2f}%')
print(f'Percentage of students with less than or equal to 20 marks in Reading: {percentage_reading_less_20:.2f}%')

print()
print("=>>> Insights From above values we get students have performed the worst in Maths "
      "Best performance is in reading section")
print("==============================================================================================================")
print()
print("4, Exploring Data (Visualization) like Average Score Distribution to Make Some Conclusion ")
print("====>>> Histogram Kernel Distribution Function (KDE)")
print()
# Create a subplot with two histograms side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first histogram with KDE for the overall average score
plt.subplot(121)
sns.histplot(data=df, x='average', bins=30, kde=True, color='g')
plt.title('Distribution of Average Scores')
plt.xlabel('Average Score')
plt.ylabel('Frequency')

# Plot the second histogram with KDE, differentiated by gender
plt.subplot(122)
sns.histplot(data=df, x='average', kde=True, hue='gender')
plt.title('Distribution of Average Scores by Gender')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
# Display the subplots
plt.tight_layout()
print("Insights 1 : Female students tend to perform well than male students");
plt.show()

print("\n")

#Lets see the effect of lunch on score of students
# Create a subplot with three histograms side by side
plt.subplots(1, 1, figsize=(8, 6))

# Plot the histogram with KDE, differentiated by lunch type (Overall)
plt.subplot(111)
sns.histplot(data=df, x='average', kde=True, hue='lunch')
plt.title('Distribution of Average Scores by Lunch (Overall female & male)')
plt.xlabel('Average Score')
plt.ylabel('Frequency')

# Display the subplots
plt.tight_layout()
print("Insights 2 : Standard lunch helps perform well in exams.");
plt.show()

#lets see parental level of education on students average score
plt.subplots(1,1,figsize=(8,6))

plt.subplot(111)
plt.title('parental level of education on students average score')
ax =sns.histplot(data=df,x='average',kde=True,hue='parental level of education')
print()
print("Insights 3 : In general parent's education don't help student perform well in exam.");
plt.show()
#lets see the effect of race/ethnicity over average score
plt.subplots(1,1,figsize=(8,6))
plt.subplot(111)
plt.title('Effect of race/ethnicity over average score')
ax =sns.histplot(data=df,x='average',kde=True,hue='race/ethnicity')
print()
print("Insights 4 : Students of group A and group B tends to perform poorly in exam.");
plt.show()
print()
#Is Race/Ethnicity impacting student's performance?
# Group the data by 'race/ethnicity'
Group_data2 = df.groupby('race/ethnicity')

# Define custom color palettes
math_palette = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
reading_palette = ['#ffb3e6', '#8cff66', '#c2f0c2', '#ffcc99', '#ffb366']
writing_palette = ['#66b3ff', '#99ff99', '#ff9999', '#c2c2f0', '#ffcc99']

# Create a subplot with three bar plots side by side
f, ax = plt.subplots(1, 3, figsize=(13, 4))

# Plot the bar plot for mean Math scores on the left
sns.barplot(x=Group_data2['math score'].mean().index, y=Group_data2['math score'].mean().values,
            palette=math_palette, ax=ax[0])
ax[0].set_title('Math score', color='#005ce6', size=20)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=10)

# Plot the bar plot for mean Reading scores in the middle
sns.barplot(x=Group_data2['reading score'].mean().index, y=Group_data2['reading score'].mean().values,
            palette=reading_palette, ax=ax[1])
ax[1].set_title('Reading score', color='#005ce6', size=20)
for container in ax[1].containers:
    ax[1].bar_label(container, color='black', size=10)

# Plot the bar plot for mean Writing scores on the right
sns.barplot(x=Group_data2['writing score'].mean().index, y=Group_data2['writing score'].mean().values,
            palette=writing_palette, ax=ax[2])
ax[2].set_title('Writing score', color='#005ce6', size=20)
for container in ax[2].containers:
    ax[2].bar_label(container, color='black', size=10)

# Display the subplots
print("Insights 5 : Group E students have scored the highest marks.")
print("             Group A students have scored the lowest marks.");
plt.show()
print()
#Effect of Test preparation on students average score Create a subplot
plt.subplot(111)

# Create a boxplot to compare the average score based on test preparation completion
sns.histplot(data=df, x='average', kde=True, hue='test preparation course')
plt.subplot(1,1,1)

# Set the title, x-axis label, and y-axis label
plt.title('Effect of Test Preparation on Average Score')
plt.xlabel('Average Score')
plt.ylabel('Frequency')

# Show the plot
print("Insights 6 : Students who have completed the Test Preparation Course have higher Average scores ")
print("             than those who haven't taken the course")
plt.show()
print("=====================================================================================================")
print()
print("5, Model building and Experiment Design")
print()


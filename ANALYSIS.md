# Analysis
* Here is the link where I did all analysis https://github.com/44-599-machine-learning-S19/machine-learning-project-vipulchandoor/blob/master/Classification.ipynb 
#### Decision Tree Classifier
* There is a notebook called Classification where we have done operations using  Decsion tree classifier.
* First we have Initial data prep section.  Read, clean and create sets.
* Created features for X and target Y. 
* The reason to make these choices are we have 3 subject marks for X and we did a average for it for value of Y 
* Made a Decision tree on X and Y and computed the matrics. 
* The results are 
1. Accuracy is  1.0
2. Precision is  1.0
3. Sensitivity is  1.0
4. F1 is  1.0
#### Support Vector Classification
* Used SVC for better results by using train sets.  
* Evaluated for the test sets and calculated the matrics as well. 
* The Results are totally different for test set and train set. 
* Train Set results are more accurtate and almost equal to 1.
* The results are 
1. Accuracy is  1.0
2. Precision is  1.0
3. Sensitivity is  1.0
4. F1 is  1.0
* Test Set results are not much accurate and they much far away from 1. 
* The results are 
1. Accuracy is  0.38
2. Precision is  0.40209480519480517
3. Sensitivity is  0.38
4. F1 is  0.35950306374500746

* This is how we train the data.
* In this we will use sklearn.model_selection model and 25% of data is used for training. 
```
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(dataset_copy, test_size=0.25, random_state = 123)
print(len(train_set))
print(len(test_set))
print(train_set.head())
print(test_set.head())
```

* Creating a feature, finding the average of all 3 courses.
* In this we are using pandas for caluclation
```
dataset["Average"] = (dataset["math score"] + dataset["reading score"] + dataset["writing score"])//3
def grades(value):
    if value > 80.0 and value <= 100.0: return "A"
    if value > 70.0 and value <= 80.0: return "B"
    if value > 60.0 and value <= 70.0: return "C"
    else: return "F"
dataset["grades"] = dataset["Average"].map(grades)
dataset.head()
plt.plot(X,Y)
plt.show()
```

* Here is the code how we find the decision tree classifier. 
* In this we are using sklearn.tree import DecisionTreeClassifier
```
from sklearn.tree import DecisionTreeClassifier
X = train_set[["math score","reading score","writing score"]]
Y = train_set["Average"]
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X,Y)
print(tree_classifier.fit(X,Y))
```

* Here is the code to find the average, gender based on test preparation course. 
* In this we are using matplotlib.pyplot and seaborn for visualization. 
```
import pandas as pd               
import numpy as np                
import matplotlib.pyplot as plt   
import seaborn as sns             

%matplotlib inline
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.barplot(x = 'test preparation course', y = 'math score',hue = 'gender', data = dataset)

plt.subplot(1,3,2)
sns.barplot(x = 'test preparation course', y = 'reading score', hue = 'gender', data = dataset)

plt.subplot(1,3,3)
sns.barplot(x = 'test preparation course', y = 'writing score',hue = 'gender', data = dataset)

plt.tight_layout()
```

* Here is the code to find the plotting of race based on the courses. 
```
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="race/ethnicity", y="math score", data=dataset, ax=axs[0],showmeans=True);
sns.boxplot(x="race/ethnicity", y="reading score", data=dataset, ax=axs[1],showmeans=True);
sns.boxplot(x="race/ethnicity", y="writing score", data=dataset, ax=axs[2],showmeans=True);
```

* Finding the relation between the parental level of study and students’ performance. 
```
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="parental level of education", y="math score", data=dataset, ax=axs[0],showmeans=True);
sns.boxplot(x="parental level of education", y="reading score", data=dataset, ax=axs[1],showmeans=True);
sns.boxplot(x="parental level of education", y="writing score", data=dataset, ax=axs[2],showmeans=True);
```

* Finding the classification of test set and train set. 
```
from sklearn.svm import SVC
X = train_set[["math score","reading score","writing score"]]
Y = train_set["Average"]
svm_classifier = SVC(kernel="rbf")
svm_classifier.fit(X,Y)
# plt.plot(X,Y)
plt.plot(kind = "bar")
plt.plot(X,Y)
plt.show()
```
```
from sklearn.svm import SVC
X = test_set[["math score","reading score","writing score"]]
Y = test_set["Average"]
svm_classifier = SVC(kernel="rbf")
svm_classifier.fit(X,Y)
```

## What challenges did you face? 
* Unable to show the results in line graph. Able to get output but unable to understand the graphs. So just changed the graphs style. 
![line](https://user-images.githubusercontent.com/31705730/56856082-eb572800-6918-11e9-8bfb-ff8c7e8eb563.png) 
#!/usr/bin/env python
# coding: utf-8

# # STA130 Homework 08
# 
# Please see the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) for the list of topics covered in this homework assignment, and a list of topics that might appear during ChatBot conversations which are "out of scope" for the purposes of this homework assignment (and hence can be safely ignored if encountered)

# <details class="details-example"><summary style="color:blue"><u>Introduction</u></summary>
# 
# ### Introduction
#     
# A reasonable characterization of STA130 Homework is that it simply defines a weekly reading comprehension assignment. 
# Indeed, STA130 Homework essentially boils down to completing various understanding confirmation exercises oriented around coding and writing tasks.
# However, rather than reading a textbook, STA130 Homework is based on ChatBots so students can interactively follow up to clarify questions or confusion that they may still have regarding learning objective assignments.
# 
# > Communication is a fundamental skill underlying statistics and data science, so STA130 Homework based on ChatBots helps practice effective two-way communication as part of a "realistic" dialogue activity supporting underlying conceptual understanding building. 
# 
# It will likely become increasingly tempting to rely on ChatBots to "do the work for you". But when you find yourself frustrated with a ChatBots inability to give you the results you're looking for, this is a "hint" that you've become overreliant on the ChatBots. Your objective should not be to have ChatBots "do the work for you", but to use ChatBots to help you build your understanding so you can efficiently leverage ChatBots (and other resources) to help you work more efficiently.<br><br>
# 
# </details>
# 
# <details class="details-example"><summary style="color:blue"><u>Instructions</u></summary>
# 
# ### Instructions
#     
# 1. Code and write all your answers (for both the "Pre-lecture" and "Post-lecture" HW) in a python notebook (in code and markdown cells) 
#     
# > It is *suggested but not mandatory* that you complete the "Pre-lecture" HW prior to the Monday LEC since (a) all HW is due at the same time; but, (b) completing some of the HW early will mean better readiness for LEC and less of a "procrastentation cruch" towards the end of the week...
#     
# 2. Paste summaries of your ChatBot sessions (including link(s) to chat log histories if you're using ChatGPT) within your notebook
#     
# > Create summaries of your ChatBot sessions by using concluding prompts such as "Please provide a summary of our exchanges here so I can submit them as a record of our interactions as part of a homework assignment" or, "Please provide me with the final working verson of the code that we created together"
#     
# 3. Save your python jupyter notebook in your own account and "repo" on [github.com](github.com) and submit a link to that notebook though Quercus for assignment marking<br><br>
# 
# </details>
# 
# <details class="details-example"><summary style="color:blue"><u>Prompt Engineering?</u></summary>
#     
# ### Prompt Engineering? 
#     
# The questions (as copy-pasted prompts) are designed to initialize appropriate ChatBot conversations which can be explored in the manner of an interactive and dynamic textbook; but, it is nonetheless **strongly recommendated** that your rephrase the questions in a way that you find natural to ensure a clear understanding of the question. Given sensible prompts the represent a question well, the two primary challenges observed to arise from ChatBots are 
# 
# 1. conversations going beyond the intended scope of the material addressed by the question; and, 
# 2. unrecoverable confusion as a result of sequential layers logial inquiry that cannot be resolved. 
# 
# In the case of the former (1), adding constraints specifying the limits of considerations of interest tends to be helpful; whereas, the latter (2) is often the result of initial prompting that leads to poor developments in navigating the material, which are likely just best resolve by a "hard reset" with a new initial approach to prompting.  Indeed, this is exactly the behavior [hardcoded into copilot](https://answers.microsoft.com/en-us/bing/forum/all/is-this-even-normal/0b6dcab3-7d6c-4373-8efe-d74158af3c00)...
# 
# </details>

# ### Marking Rubric (which may award partial credit)
# - [0.1 points]: All relevant ChatBot summaries [including link(s) to chat log histories if you're using ChatGPT] are reported within the notebook
# - [0.2 points]: Well-communicated and sensible answers for Question "2"
# - [0.2 points]: Correct code and well-communicated correct answer for Question "4" 
# - [0.2 points]: Correct calculations for requested metrics in Question "6" 
# - [0.3 points]: Correct and well-communicated explanation of differences for Question "7" 
# <!-- - [0.1 points]: Written submission evaluation and enagement confirmation with ChatBot summaries for "8", "10"-->
# 
# 

# ## "Pre-lecture" HW [*completion prior to next LEC is suggested but not mandatory*]

# 
# ### 1. Start a ChatBot session to understand what a *Classification Decision Tree* is: (a) ask the ChatBot to describe the type of problem a *Classification Decision Tree* addresses and provide some examples of real-world applications where this might be particularly useful, and then (b) make sure you understand the difference between how a *Classification Decision Tree* makes *(classification) predictions* versus how *Multiple Linear Regression* makes *(regression) predictions*<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _The first part (a) of this question is looking for you to understand the difference between **classification** and **regression**. The second part (b) of the questions is looking for a fairly high level understanding of the general nature of a decision tree and how it is based on making sequential decisions down the *nodes* of *tree* in order to eventually make a final prediction. This part (b) is essentially the **Classification Decision Tree** analog of "explain how the **linear form** makes a prediciton in **Multiple Linear Regression** generally speaking"; namely,"explain how the **tree** makes a prediciton in a **Classification Decision Tree** generally speaking"._
# > 
# > _**If you're struggling with this, it would probably be most helpful to go search for and some images of example decision trees to look at!**_
# > 
# > - _You may be beginning to realize or will nonetheless eventually come to understand that the sequential decisions at each stage of the **Decision Tree** are **interactions** (in the same manner as **interactions** in **Multiple Linear Regression**.  Once you start to see that and it's making sense to you then you'll increasingly appreciate how **complex** **Decision Tree** models can be, even though they're pretty simple to understand if you just look at one._
# >
# > ---
# >    
# > _When using chatbots, it's often more effective (and enjoyable) to ask concise, single questions rather than presenting complex, multi-part queries. This approach can help in obtaining clearer and more specific responses (that might be more enjoyable to interact with). You can always ask multi-part questions as a series of additional sequential questions. With this approach, chatbots may not automatically reiterate previously explained concepts. So if you need a refresher or further explanation on a topic discussed earlier, just explicitly request during follow-up interactions._
# > 
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details>

# In[2]:


# a) A Classification Decision Tree solves classification problems, 
# where the goal is to assign data points to specific categories or 
# labels.
# Examples of real-world applications:
# - Diagnosing diseases in health care. 
# - Customer segmentation in marketing.
# - Fraud detection in finance.
# - Predicting student outcomes in education.
# - Recommending products in e-commerce.

# b) Classification Decision Tree:
# - Outputs categorical labels by following a tree-like structure where
# data is split based on feature values.
# - Example: Deciding if a transaction is "Fraudlent" or "Not 
# Fraudelent".

# Multiple Linear Regression:
# - Outputs continous numerical values using a linear equation that 
# combines features.
# - Example: Predicting the price of a car based on mileage, age, and
# condition.

# Key Difference.
# - A decision tree predicts categories, while linear regression 
# predicts numbers.


# ### 2. Continue your ChatBot session and explore with your ChatBot what real-world application scenario(s) might be most appropriately addressed by each of the following *metrics* below: provide your answers and, in your own words, *concisely explain your rationale for your answers.*<br>
# 
# 
# 1. **Accuracy** measures the proportion of true results (both true positives and true negatives) in the population.
# 
# $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
#     
# 2. **Sensitivity** measures the proportion of actual positives that are correctly identified.
# 
# $$\text{Sensitivity} = \frac{TP}{TP + FN}$$
# 
# 3. **Specificity** measures the proportion of actual negatives that are correctly identified.
# 
# $$\text{Specificity} = \frac{TN}{TN + FP}$$
# 
# 4. **Precision** measures the proportion of positive identifications that were actually correct.
# 
# $$\text{Precision} = \frac{TP}{TP + FP}$$
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _This question could be restated as, "Give examples of real-world examples where each of these **metrics** would be particularly useful."_
# >
# > _The primary objective here is to understand the distinction between each of these **metrics**. The secondary objective is to notice how the nature of the decision-making that each of these **metrics** most naturally supports is very distinct, ideally based on identifying memorable examples that really distinguish between the **metrics**._
# >
# > - _Have a look at this (greatly expanded) handy list of additional metrics, formulas, and synonyms at the following [wikipedia page](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) if you want this to get real crazy real fast._
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details>

# In[4]:


# 1)
# - Best Used When: Errors (false positives and flase negatives) have 
# similar costs.
# - Example Scenario: Spam email detection; Rationale: Both
# misclassifying spam as legitimate and legitimate emails as spam are 
# equally undeseriable.

# 2) 
# - Best Used When: False negatives are very costly.
# - Example Scenario: Cancer diagnosis; Rationale: missing an actual
# cancer case (false negative) can lead to severe consequences, so
# identifying postives is critical.

# 3) 
# - Best Used When: False negatives are very costly.
# - Example Scenario: Cancer diagnosis; Rationale: Incorrectly flagging
# legitimate transactions (false positives) can inconvenience 
# customers and create inefficiencies.

# 4) 
# - Best Used When: False positives need to be minimized for reliable
# predictions.
# - Example Scenario: Pedestrian detection in autonomous cars; 
# Rationale: If the system predicts a pedestrian, it must be highly 
# condident to avoud u


# ### 3. Explore the amazon books dataset, seen previously at the start of the semester, providing some initital standard *exploratory data analysis (EDA)* and data summarization after pre-processing the dataset to meet the requirements below<br>
# 
#  1. remove `Weight_oz`, `Width`, and `Height` 
#  2. drop all remaining rows with `NaN` entries 
#  3. set `Pub year` and `NumPages` to have the type `int`, and `Hard_or_Paper` to have the type `category`
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _`NaN` entries can't be used in their raw form with the `scikit-learn` methodologies, so we do need to remove them to proceed with our analyses._
# >     
# > _Only remove rows with `NaN` entries once you've subset to the columns you're interested in. This will minimize potentially unnecessary data loss..._
# >
# > _It would be possible to consider imputing missing data to further mitigate data loss, but the considerations for doing so are more advanced than the level of our course, so we'll not consider that for now._ 
# 
# </details>

# In[1]:


import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/point0five/STA130_F23/main/Data/amazonbooks.csv"
ab = pd.read_csv(url, encoding="ISO-8859-1")

# Step 1: Remove specified columns
ab_reduced = ab.drop(columns=['Weight_oz', 'Width', 'Height'])

# Step 2: Drop rows with NaN entries
ab_reduced_noNaN = ab_reduced.dropna()

# Step 3: Change data types
ab_reduced_noNaN['Pub year'] = ab_reduced_noNaN['Pub year'].astype(int)
ab_reduced_noNaN['NumPages'] = ab_reduced_noNaN['NumPages'].astype(int)
ab_reduced_noNaN['Hard_or_Paper'] = ab_reduced_noNaN['Hard_or_Paper'].astype('category')

# Display the cleaned dataset
print(ab_reduced_noNaN.head())

# Perform basic EDA
# Summary statistics
print(ab_reduced_noNaN.describe())
print(ab_reduced_noNaN.info())

# Distribution of numerical columns
ab_reduced_noNaN[['Pub year', 'NumPages']].hist()

# Value counts for the categorical column
print(ab_reduced_noNaN['Hard_or_Paper'].value_counts())

# Correlations (for numerical columns)
print(ab_reduced_noNaN.corr())


# ### 4. Create an 80/20 split with 80% of the data as a training set *ab_reduced_noNaN_train* and 20% of the data testing set  *ab_reduced_noNaN_test* using either *df.sample(...)* as done in TUT or using *train_test_split(...)* as done in the previous HW, and report on how many observations there are in the training data set and the test data set.<br><br>Tell a ChatBot that you are about to fit a "scikit-learn" *DecisionTreeClassifier* model and ask what the two steps given below are doing; then use your ChatBots help to write code to "train" a classification tree *clf* using only the *List Price* variable to predict whether or not a book is a hard cover or paper back book using a *max_depth* of *2*; finally use *tree.plot_tree(clf)* to explain what *predictions* are made based on *List Price* for the fitted *clf* model
# 
# ```python
# y = pd.get_dummies(ab_reduced_noNaN["Hard_or_Paper"])['H']
# X = ab_reduced_noNaN[['List Price']]
# ```
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _To complete the final 80/20 split of the **observations** in a reproducible way, set a "random seed"._ 
# > 
# > - _A single **observation** consists of all the measurements made on a single entity, typically corresponding to a row of a data frame. In **Machine Learning**, a collection of values of interest measured for a single entity is called a "vector" and so the **observation** is referred to as a **vector**_.
# >    
# > _Asking the ChatBot about "DecisionTreeClassifier .fit(...)" can be helpful here..._
# > 
# > _Should you use the "ab_reduced_noNaN" data, or the "ab_reduced_noNaN_train" data, or the "ab_reduced_noNaN_test" data to initially fit the classification tree? Why?_
# >    
# > _You can visualize your decision tree using the `tree.plot_tree(clf)` function shown in the `sklearn` documentation [here](
# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#what-is-the-values-array-used-here) and [here](https://scikit-learn.org/stable/modules/tree.html); but, to make it more immediately readible it might be better to use `graphviz`, which is demonstrated in the `sklearn` documentation [here](https://scikit-learn.org/stable/modules/tree.html#alternative-ways-to-export-trees)_ 
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details>

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the Data
url = "https://raw.githubusercontent.com/point0five/STA130_F23/main/Data/amazonbooks.csv"
ab = pd.read_csv(url, encoding="ISO-8859-1")

# Preprocessing steps
ab_reduced = ab.drop(columns=['Weight_oz', 'Width', 'Height'])
ab_reduced_noNaN = ab_reduced.dropna()
ab_reduced_noNaN['Pub year'] = ab_reduced_noNaN['Pub year'].astype(int)
ab_reduced_noNaN['NumPages'] = ab_reduced_noNaN['NumPages'].astype(int)
ab_reduced_noNaN['Hard_or_Paper'] = ab_reduced_noNaN['Hard_or_Paper'].astype('category')

# Define target variable (y) and predictor variable (X)
y = pd.get_dummies(ab_reduced_noNaN["Hard_or_Paper"])['H']  # Binary classification: Hardcover ('H') = 1, Paperback ('P') = 0
X = ab_reduced_noNaN[['List Price']]  # Use List Price as the feature

# Step 2: Perform an 80/20 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Report the number of observations
print(f"Training set observations: {len(X_train)}")
print(f"Testing set observations: {len(X_test)}")

# Step 3: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Visualize the Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=["List Price"], class_names=["Paperback", "Hardcover"], filled=True)
plt.show()

# Nodes in the tree: Each node corresponds to a decision threshold on 
# the "List Price".

# Leaves: Each leaf represents a final classification (Paperback or 
# Hardcover) based on the majority class in that subset of data.

# Depth: The max_depth=2 ensures the tree will have at most 2 splits, 
# creating clear and interpretable predictions.


# <details class="details-example"><summary style="color:blue"><u>Continue now...?</u></summary>
# 
# ### Pre-lecture VS Post-lecture HW
# 
# Feel free to work on the "Post-lecture" HW below if you're making good progress and want to continue: for **HW 08** this could be reasonable because, as you'll see, the process of creating and using **classification decision trees** is quite similar to the process for creating and using **multiple linear regression** models. There are differences of course, such as how there is **coefficient hypothesis testing** in **multiple linear regression** and **confusion matrices** in **classification decision trees**, and so on. But you would very likely be able to leverage the silarities to make a lot of progress with **classification decision trees** based on your experience with **multiple linear regression**.
#     
# *The benefits of continue would are that (a) Consolidate the knowledge already learned and integrate it comprehensively. (b) Let you build experience interacting with ChatBots (and understand their strengths and limitations in this regard)... it's good to have sense of when using a ChatBot is the best way to figure something out, or if another approach (such as course provided resources or a plain old websearch for the right resourse) would be more effective*
#     
# </details>    

# ## "Post-lecture" HW [*submission along with "Pre-lecture" HW is due prior to next TUT*]
# 

# ### 5. Repeat the previous problem but this time visualize the *classification decision tree* based on the following specifications below; then explain generally how predictions are made for the *clf2* model<br>
# 
# 1. `X = ab_reduced_noNaN[['NumPages', 'Thick', 'List Price']]`
# 2. `max_depth` set to `4`
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > - _Use the same **train/test split** dataset used so far_
# > - _Train the **classification decision tree** `clf2` using **predictor variables** `NumPages`, `Thick` and `List Price`_ 
# > - _Again **predict** whether or not a book is hard cover book or a paper back book_
# > - _You can visualize your decision tree using the `tree.plot_tree(clf)` function shown in the `sklearn` documentation [here](
# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#what-is-the-values-array-used-here) and [here](https://scikit-learn.org/stable/modules/tree.html); but, to make it more immediately readible it might be better to use `graphviz`, which is demonstrated in the `sklearn` documentation [here](https://scikit-learn.org/stable/modules/tree.html#alternative-ways-to-export-trees)_
# >
# > _If you are interested in how to find the best `max_depth` for a tree, ask ChatBot about "GridSearchCV"_
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Define the new feature set (X) and target variable (y)
X = ab_reduced_noNaN[['NumPages', 'Thick', 'List Price']]
y = pd.get_dummies(ab_reduced_noNaN["Hard_or_Paper"])['H']  # Target variable: Hardcover (H = 1, P = 0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier with max_depth=4
clf2 = DecisionTreeClassifier(max_depth=4, random_state=42)

# Train the classifier on the training data
clf2.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf2, feature_names=['NumPages', 'Thick', 'List Price'], class_names=['Paperback', 'Hardcover'], filled=True)
plt.show()

# Feature Importance: Predictions depend on the interaction of NumPages,
# Thick, and List Price.

# Tree Depth: A deeper tree (max_depth = 4) enables more detailed 
# classification by considering more conditions before making a 
# prediction.

# Thresholds: At each level, the tree uses a threshold on one feature to
# split the data. The thresholds are optimized to best seperate the 
# classes at that point.

# Outcome: The tree uses combinations of page numbers, thickness, and 
# price to classify whether a book is a *Hardcover* or *Paperback*.


# ### 6. Use previously created *ab_reduced_noNaN_test* to create confusion matrices for *clf* and *clf2*. Report the sensitivity, specificity and accuracy for each of the models<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _Hopefully you immediately thought to ask ChatBot to help you with this problem, but if you did you should take time to make sure you're clear about the key components of what the ChatBot is providing for you. You might want to know_
# > - _what is a "positive" and what is a "negative"_
# > - _how to read an `sklearn` confusion matrix_
# > - _what leads to TP, TN, FP, and FN_
# > - _whether `y_true` or `y_pred` go first in the `confusion_matrix` function_   
# >
# > _Have the visualizations you make use decimal numbers with three signifiant digits, such as `0.123` (and not as percentages like `12.3%`), probably based on `np.round()`_
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#        
# </details>

# In[9]:


from sklearn.metrics import confusion_matrix, accuracy_score

# Predict the test data using both models
y_test = pd.get_dummies(ab_reduced_noNaN_test["Hard_or_Paper"])['H']
X_test_clf = ab_reduced_noNaN_test[['List Price']]
X_test_clf2 = ab_reduced_noNaN_test[['NumPages', 'Thick', 'List Price']]

# Predictions
y_pred_clf = clf.predict(X_test_clf)
y_pred_clf2 = clf2.predict(X_test_clf2)

# Confusion Matrices
cm_clf = confusion_matrix(y_test, y_pred_clf)
cm_clf2 = confusion_matrix(y_test, y_pred_clf2)

# Extracting Metrics for clf
TP_clf, FN_clf, FP_clf, TN_clf = cm_clf[1, 1], cm_clf[1, 0], cm_clf[0, 1], cm_clf[0, 0]
sensitivity_clf = TP_clf / (TP_clf + FN_clf)
specificity_clf = TN_clf / (TN_clf + FP_clf)
accuracy_clf = (TP_clf + TN_clf) / cm_clf.sum()

# Extracting Metrics for clf2
TP_clf2, FN_clf2, FP_clf2, TN_clf2 = cm_clf2[1, 1], cm_clf2[1, 0], cm_clf2[0, 1], cm_clf2[0, 0]
sensitivity_clf2 = TP_clf2 / (TP_clf2 + FN_clf2)
specificity_clf2 = TN_clf2 / (TN_clf2 + FP_clf2)
accuracy_clf2 = (TP_clf2 + TN_clf2) / cm_clf2.sum()

# Print Results
print("Metrics for clf (Single Feature):")
print(f"Sensitivity: {sensitivity_clf:.2f}, Specificity: {specificity_clf:.2f}, Accuracy: {accuracy_clf:.2f}")
print("\nMetrics for clf2 (Multiple Features):")
print(f"Sensitivity: {sensitivity_clf2:.2f}, Specificity: {specificity_clf2:.2f}, Accuracy: {accuracy_clf2:.2f}")

# Results:
# Sensitivity:
# - Measures the ability of the model to correctly predict Hardcover 
# books.
# - Higher sensitivity means fewer Hardcover Books are missclassified
# as Paperback.

# Specificty:
# - Measures the ability of the model to correctly predict Paperback
# books.
# - Higher specificity means fewer Paperback books are misclassified as
# Hardcover.

# Accuracy:
# - Measures the overall correctness of the predictions for both 
# classes.


# ### 7. Explain in three to four sentences what is causing the differences between the following two confusion matrices below, and why the two confusion matrices above (for *clf* and *clf2*) are better<br>
# 
# ```python
# ConfusionMatrixDisplay(
#     confusion_matrix(ab_reduced_noNaN_train.life_exp_good, 
#                      clf.predict(ab_reduced_noNaN_train[['List Price']]), 
#                      labels=[0, 1]), display_labels=["Paper","Hard"]).plot()
# ConfusionMatrixDisplay(
#     confusion_matrix(ab_reduced_noNaN_train.life_exp_good, 
#                      clf.predict(
#                          ab_reduced_noNaN_train[['NumPages','Thick','List Price']]), 
#                      labels=[0, 1]), display_labels=["Paper","Hard"]).plot()
# ```
# 
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#     
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details>

# In[10]:


# The differences between the two confusion matrices arise because the 
# first model (clf) uses only List Price as the feature for prediction,
# while the second model (clf2) incorporates multiple features such as:
# NumPages, Thick, and List Price. By using more features, the second
# model has more information to split the data, which can lead to 
# improved classification performance if those additional features are
# meaningful predictors. The confusion matrices for clf and clf2 on the 
# test set are better because they evaluate how well the models 
# generalize to unseen data, rather than merely acssessing performance 
# on the training set. The confusion matrices for the training set are
# likely to show lower errors because the model has already seen the
# data during training, potentially leading to overfitting. By contrast,
# the test set matrices provide a more accurate measure of the models' 
# true performance in real-world scenarios.


# 
# ### 8. Read the paragraphs in *Further Guidance* and ask a ChatBot how to visualize *feature Importances* available for *scikit-learn* *classification decision trees*; do so for *clf2*;  and use *.feature_names_in_* corresponding to *.feature_importances_* to report which *predictor variable* is most important for making predictions according to *clf2*<br>
# 
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#   
# > The way a **classification decision tree** is fit is that at each step in the construction process of adding a new **decision node splitting rule** to the current **tree structure**, all possible **decision rules** for all possible **predictor variables** are considered, and the combination that improves the **prediction** the most (as measured by the criterion of either "Gini impurity" or "Shannon entropy") and in accordance with the rules of the decision tree (such as the `max_depth` argument) is added to the **classification decision tree**.  Thus overall "criterion" noted above improves with each new **decision node splitting rule**, so the improvement can thus be tracked and the improvement contributions attributed to the **feature** upon which the **decision node splitting rule** is based.  This means the relative contribution of each **predictor variable** to the overall explanatory power of the model can be calculated, and this is what the `.feature_importances_` attribute does. 
# >
# > Compared to the simplicity of understanding how different **covariates** contribute towards the final **predicted values** of **multiple linear regression models** (by just reading off the equation to see how predictions work), the the complexity of how all the different **features** interact and combine to together to create the final **predictions** from **classification decision trees** can be staggering. But the so-called **feature importance** heuristics allows us to judge how relatively important the overall contributions from different features are in the final decision tree predictions. Now we just need to be sure we're not **overfitting** our **classification decision trees** since they can be so **complex**. Fortunately, the "GridSearchCV" methodology mentioned in regards to finding the best `max_depth` setting for a tree is going to provide a general answer to the challenge of complexity and **overfitting** in **machine learning models** that is not too hard to understand (and which you might already have some guesses or a hunch about). 
# > 
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#        
# </details>

# In[12]:


import matplotlib.pyplot as plt

# Get feature importances from clf2
feature_importances = clf2.feature_importances_

# Get the feature names corresponding to the importance values
feature_names = clf2.feature_names_in_

# Plot feature importances
plt.figure(figsize=(8, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances for clf2")
plt.show()

# Identify the most important predictor variable
most_important_feature = feature_names[feature_importances.argmax()]
most_important_importance = feature_importances.max()

print(f"The most important predictor variable is: {most_important_feature}")
print(f"Importance score: {most_important_importance:.4f}")

# 1. Visualization:
# - A bar chart will display the relative contributions of NumPages, 
# Thick, and List Price to the tree's overall predictive power.
# - The longer the bar, the more important the variable.

# 2. Identification of the Most Important Variable:
# - The variable with the highest importance score is the one the model 
# relied on the most for making predictions.
# - The .feature_importances_ attribute directly measures this 
# importance.

# -> If List Price is the most important feature, it means it provides
# the strongest signal for seperating Hardcover from Paperback.

# -> If NumPages or Thick has a high importance, it indicates these 
# features provide additional explanatory power beyond price.


# In[13]:


# In Linear Regression, the coefficients represent the strength and 
# direction of the relationship between each predictor variable and the
# target variable, with larger absolute values indicating more 
# influence. In decision trees, feature importance quantifies how much 
# a feature contributes to reducing the impurity (e.g. Gini or entropy)
# across all splits in the tree, with higher importance indicating a 
# feature's greater role in making accurate predictions. While linear
# regression coefficiants represent the direct, linear relationship
# between predictors and the target, decision tree feature importances
# reflect how much each feature helps to improve the model's
# performance through non-linear splits in the tree structure.


# ### 10. Have you reviewed the course wiki-textbook and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br>
#   
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# >  _Here is the link of [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) in case it gets lost among all the information you need to keep track of_  : )
# > 
# > _Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)_
#     
# </details>

# In[ ]:


# yessirrrr


# # Recommended Additional Useful Activities [Optional]
# 
# The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course 
# 
# <br>
# <details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>
# 
# ### Ethical Professionalism Considerations
# 
# - Discuss with a ChatBox about consent and data collection for training models.
#     - Discuss the ethics of data collection for training decision trees, particularly the need for informed consent when personal data is involved.
#     - Evaluate the role of regulatory frameworks in ensuring ethical data collection practices.
# - Discuss with a ChatBox about accountability in automated decision-making.
#     - Address the challenges of holding systems and their developers accountable when decision trees lead to adverse outcomes.
#     - Explore legal and ethical frameworks for responsibility when automated decisions go wrong.
# - Discuss with a ChatBox about transparency and explainability in classification models.
#     - Discuss the importance of model transparency, particularly when using decision trees in sectors like healthcare or criminal justice.
#     - Explore methods to enhance the explainability of decision trees, such as visualization techniques and simplified decision paths.
# - Discuss with a ChatBox about impact of misclassifications in critical applications.
#     - Examine the consequences of false positives and false negatives in decision tree outcomes, using confusion matrices to highlight these issues.
#     - Discuss ethical responsibilities when deploying classifiers in high-stakes fields like medicine or law enforcement.
#     
# </details>    
# 
# <details class="details-example"><summary style="color:blue"><u>Current Course Project Capability Level</u></summary>
# 
# **Remember to abide by the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) at all times.**
# 
# Information about the course project is available on the course github repo [here](https://github.com/pointOfive/stat130chat130/tree/main/CP), including a draft [course project specfication](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F23_course_project_specification.ipynb) (subject to change). 
# - The Week 01 HW introduced [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb), and the [available variables](https://drive.google.com/file/d/1ISVymGn-WR1lcRs4psIym2N3or5onNBi/view). 
# - Please do not download the [data](https://drive.google.com/file/d/1mbUQlMTrNYA7Ly5eImVRBn16Ehy9Lggo/view) accessible at the bottom of the [CSCS](https://casch.org/cscs) webpage (or the course github repo) multiple times.
#     
# 
# > ### NEW DEVELOPMENT<br>New Abilities Achieved and New Levels Unlocked!!!    
# > 
# > And with that, ALL LEVELS unlocked! 
# >
# > CONGRATS, YOU LEGENDS! 🎉
# >
# > You’ve battled through the wild jungles of deadlines, defeated the mighty Homework Beasts, and climbed the towering Mount Procrastination. And guess what? YOU MADE IT TO THE TOP! 🏔️
# > 
# > Take a bow, grab a treat, and enjoy the sweet, sweet taste of freedom(**just for now , because you still have to finish the project! But you are almost done!**). You’ve earned it. Now go out there and celebrate like the absolute rockstars you are! 🌟💪
# >
# 
#     
# ### Current Course Project Capability Level    
#     
# I mean, the **course project** is basically, like, essentially now.
#     
# - Will you be doing any **classification decision trees** stuff for the course project?
#     - You could consider making some [partial dependency plots](https://scikit-learn.org/stable/modules/partial_dependence.html) if so...
#     - those might provide an interesting analysis in addition to **tree structure** visualizations, **confusion matrices**, **feature importances**. and the standard "in-sample versus out-of-sample" **train-test validation** analysis that would be expected in a **machine learning context**
#     
# - You could see if there are any interesting columns that might make for a potentially interesting **classification decision tree** analysis
#     - You wouldn't have to though...
#     - But if you did you'd want to be able to articulate and explain why what you're doing with **classification decision trees** is appropriate and enlightening
# 
# - Anyway, I guess that just leaves reviewing all the statistical techniques covered in STA130, and considering integrating them holistically into your project!
#     
# </details>        

# In[ ]:





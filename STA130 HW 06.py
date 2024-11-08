#!/usr/bin/env python
# coding: utf-8

# # STA130 Homework 06
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
# 
# - [0.1 points]: All relevant ChatBot summaries [including link(s) to chat log histories if you're using ChatGPT] are reported within the notebook
# - [0.2 points]: Evaluation of correctness and clarity in written communication for Question "3"
# - [0.2 points]: Evaluation of correctness and clarity in written communication for Question "4"
# - [0.2 points]: Evaluation of submitted work and conclusions for Question "9"
# - [0.3 points]: Evaluation of written communication of the "big picture" differences and correct evidence assessement for Question "11"
# 

# ## "Pre-lecture" versus "Post-lecture" HW? 
# 
# - _**Your HW submission is due prior to the Nov08 TUT on Friday after you return from Reading Week; however,**_
# - _**this homework assignment is longer since it covers material from both the Oct21 and Nov04 LEC (rather than a single LEC); so,**_
# - _**we'll brake the assignment into "Week of Oct21" and "Week of Nov04" HW and/but ALL of it will be DUE prior to the Nov08 TUT**_
# 

# ## "Week of Oct21" HW [*due prior to the Nov08 TUT*]

# ### 1. Explain the theoretical Simple Linear Regression model in your own words by describing its components (of predictor and outcome variables, slope and intercept coefficients, and an error term) and how they combine to form a sample from normal distribution; then, create *python* code explicitly demonstrating your explanation using *numpy* and *scipy.stats* <br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _Your answer can be quite concise and will likely just address the "mathematical" and "statistical" aspects of the process of a **Simple Linear Model** specification, perhaps giving an intuitive interpretation summary of the result as a whole_
# >   
# > - _Your code could be based on values for `n`, `x`, `beta0`, `beta1`, and `sigma`; and, then create the `errors` and `Y`_
# > 
# > - _The predictors $x_i$ can be fixed arbitrarily to start the process (perhaps sampled using `stats.uniform`), and they are conceptually different from the creation of **error** (or **noise**) terms $\epsilon_i$ which are sampled from a **normal distribution** (with some aribtrarily *a priori* chosen **standard deviation** `scale` parameter $\sigma$) which are then combined with $x_i$ through the **Simple Linear Model** equation (based on aribtrarily *a priori* chosen **slope** and **intercept coefficients**) to produce the $Y_i$ outcomes_
# > 
# > - _It should be fairly easy to visualize the "a + bx" line defined by the **Simple Linear Model** equation, and some **simulated** data points around the line in a `plotly` figure using the help of a ChatBot_
# > 
# > _If you use a ChatBot (as expected for this problem), **don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)**_
# >
# > 
# > _**Question Scope Warning:** Be careful when using a ChatBot to help you with creating an example dataset and coding up a visualization though, **because it might suggest creating (and visualizing) a fitted model for to your data (rather than the theoretical model); but, this is not what this question is asking you to demonstrate**. This question is not asking about how to produce a fitted **Simple Linear Regression** model or explain how model **slope** and **intercept coefficients** are calculated (e.g., using "ordinary least squares" or analytical equations to estimate the **coefficients**  for an observed dataset)._
# > 
# > ```python
# > # There are two distinct ways to use `plotly` here
# >
# > import plotly.express as px
# > px.scatter(df, x='x',  y='Y', color='Data', 
# >            trendline='ols', title='Y vs. x')
# >        
# > import plotly.graph_objects as go
# > fig = go.Figure()
# > fig.add_trace(go.Scatter(x=x, y=Y, mode='markers', name='Data'))
# > 
# > # The latter is preferable since `trendline='ols'` in the former 
# > # creates a fitted model for the data and adds it to the figure
# > # and, again, THAT IS NOT what this problem is asking for right now
# > ```
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
# </details>
# 

# In[1]:


# In Simple Linear Regression, we try to understand the relationship
# of 2 varaibles, X (predictor variable) and Y (outcome variable). This
# relationship is assumed to be a straight line, represented by the 
# equation: Y = B0 + b1X + e (epsilon) where b0 is the intercept, b1 is
# the slope, and epsilon is the error term, representing factors not
# explained by X. I like to think of this equatiion as the same equation
# as the slope formula y = mx + b, where in this case slope is being added
# and epsilon is the x term which the term changing the entire equation
# if that makes sense?

# code:
import numpy as np
import matplotlib.pyplot as plt

# Define the true values for intercept and slope
beta_0 = 2  # intercept
beta_1 = 3  # slope
sigma = 1   # standard deviation of the error

# Generate 50 random X values between 0 and 10
X = np.linspace(0, 10, 50)

# Generate random error terms (normal distribution with mean 0 and std deviation 'sigma')
epsilon = np.random.normal(0, sigma, size=len(X))

# Calculate the Y values using the equation Y = beta_0 + beta_1 * X + error
Y = beta_0 + beta_1 * X + epsilon

# Plot the data
plt.scatter(X, Y, label='Data', color='blue')
plt.plot(X, beta_0 + beta_1 * X, label='True Line', color='red')  # The true line without noise
plt.xlabel('X (Predictor)')
plt.ylabel('Y (Outcome)')
plt.legend()
plt.show()


# ### 2. Use a dataset simulated from your theoretical Simple Linear Regression model to demonstrate how to create and visualize a fitted Simple Linear Regression model using *pandas* and *import statsmodels.formula.api as smf*<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > - _Combine the **simulated** `x` and `Y` into a `pandas` data frame object named `df` with the column names "x" and "Y"_
# > 
# > - _Replace the inline question comments below with their answers (working with a ChatBot if needed)_
# >
# > ```python
# > import statsmodels.formula.api as smf  # what is this library for?
# > import plotly.express as px  # this is a ploting library
# >
# > # what are the following two steps doing?
# > model_data_specification = smf.ols("Y~x", data=df) 
# > fitted_model = model_data_specification.fit() 
# >
# > # what do each of the following provide?
# > fitted_model.summary()  # simple explanation? 
# > fitted_model.summary().tables[1]  # simple explanation?
# > fitted_model.params  # simple explanation?
# > fitted_model.params.values  # simple explanation?
# > fitted_model.rsquared  # simple explanation?
# >
# > # what two things does this add onto the figure?
# > df['Data'] = 'Data' # hack to add data to legend 
# > fig = px.scatter(df, x='x',  y='Y', color='Data', 
# >                  trendline='ols', title='Y vs. x')
# >
# > # This is essentially what above `trendline='ols'` does
# > fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
# >                 line=dict(color='blue'), name="trendline='ols'")
# > 
# > fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
# > ```
# >
# > _The plotting here uses the `plotly.express` form `fig.add_scatter(x=x, y=Y)` rather than the `plotly.graph_objects` form `fig.add_trace(go.Scatter(x=x, y=Y))`. The difference between these two was noted in the "Further Guidance" comments in the previous question; but, the preference for the former in this case is because `px` allows us to access `trendline='ols'` through `px.scatter(df, x='x',  y='Y', trendline='ols')`_
# >
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_      
# 
# </details>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Step 1: Simulate data
np.random.seed(42)

# Define parameters
beta_0 = 2  # intercept
beta_1 = 3  # slope
sigma = 1   # standard deviation of the error term

# Generate predictor variable X (50 random points between 0 and 10)
X = np.linspace(0, 10, 50)

# Generate random error terms (normal distribution)
epsilon = np.random.normal(0, sigma, size=len(X))

# Create the outcome variable Y
Y = beta_0 + beta_1 * X + epsilon

# Step 2: Create a DataFrame
data = pd.DataFrame({'X': X, 'Y': Y})

# Step 3: Fit the Simple Linear Regression model using statsmodels
# Using formula API to specify the regression model: Y ~ X
model = smf.ols(formula='Y ~ X', data=data).fit()

# Step 4: Print the model summary
print(model.summary())

# Step 5: Plot the original data and the fitted regression line
plt.scatter(data['X'], data['Y'], label='Data', color='blue')
plt.plot(data['X'], model.fittedvalues, label='Fitted Line', color='red', linewidth=2)
plt.xlabel('X (Predictor)')
plt.ylabel('Y (Outcome)')
plt.title('Simple Linear Regression Fit')
plt.legend()
plt.show()


# ### 3. Add the line from Question 1 on the figure of Question 2 and explain the difference between the nature of the two lines in your own words; *but, hint though: simulation of random sampling variation*<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _This question is effectively asking you to explain what the combined code you produced for Questions 1 and 2 is trying to demonstrate overall. If you're working with a ChatBot (as expected), giving these two sets of code as context, and asking what the purpose of comparing these lines could be would be a way to get some help in formulating your answer_
# > 
# > _The graphical visualization aspect of this question could be accomplished by appending the following code to the code provided in Question 2._
# > 
# > ```python
# > # what does this add onto the figure in constrast to `trendline='ols'`?
# > x_range = np.array([df['x'].min(), df['x'].max()])
# > # beta0 and beta1 are assumed to be defined
# > y_line = beta0 + beta1 * x_range
# > fig.add_scatter(x=x_range, y=y_line, mode='lines',
# >                 name=str(beta0)+' + '+str(beta1)+' * x', 
# >                 line=dict(dash='dot', color='orange'))
# >
# > fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
# > ```
# > 
# > _The right way to interactively "see" the answer to this question is to repeatedly create different dataset **simulations** using your theoretical model and the corresponding fitted models, and repeatedly visualize the data and the two lines over and over... this would be as easy as rerunning a single cell containing your simulation and visualization code..._
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
# </details>

# In[3]:


# Green-Dashed Line (True Line): represents the idealized relationship
# between Y and X, assuming no random error or varaibility in the data.

# Red-Solid Line (Fitted Line): this is the result of the linear
# regression model we applied ot the data. It includes the random 
# variation I mentioned in the first question (epsilon), which is why
# the data deviates from the true line.

# the difference between the 2 lines help illustrate the impact of
# random sampling variation since in the real world, no data can 
# perfectly matha. theoretical model due to there always being some 
# uncontrollable factors. And the factors are the reason why the line
# deviates from the true line allbeit slightly.

# Updated Code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Step 1: Simulate data
np.random.seed(42)

# Define parameters
beta_0 = 2  # intercept
beta_1 = 3  # slope
sigma = 1   # standard deviation of the error term

# Generate predictor variable X (50 random points between 0 and 10)
X = np.linspace(0, 10, 50)

# Generate random error terms (normal distribution)
epsilon = np.random.normal(0, sigma, size=len(X))

# Create the outcome variable Y
Y = beta_0 + beta_1 * X + epsilon

# Step 2: Create a DataFrame
data = pd.DataFrame({'X': X, 'Y': Y})

# Step 3: Fit the Simple Linear Regression model using statsmodels
model = smf.ols(formula='Y ~ X', data=data).fit()

# Step 4: Print the model summary
print(model.summary())

# Step 5: Plot the original data, true line, and fitted regression line
plt.scatter(data['X'], data['Y'], label='Data', color='blue')
plt.plot(data['X'], beta_0 + beta_1 * data['X'], label='True Line (No Noise)', color='green', linestyle='--', linewidth=2)
plt.plot(data['X'], model.fittedvalues, label='Fitted Line', color='red', linewidth=2)
plt.xlabel('X (Predictor)')
plt.ylabel('Y (Outcome)')
plt.title('Simple Linear Regression Fit vs. True Line')
plt.legend()
plt.show()


# ### 4. Explain how *fitted_model.fittedvalues* are derived on the basis of *fitted_model.summary().tables[1]* (or more specifically  *fitted_model.params* or *fitted_model.params.values*)<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _The previous questions used code to explore the distinction between theoretical (true) $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \;[\text{where } \epsilon_i \sim \mathcal{N}(0, \sigma)]\;$ and fitted (estimated) $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ **Simple Linear Regression** models_
# >
# > _This question asks you to explicitly illustrate how the the latter "in sample predictions" of the fitted **Simple Linear Regression** model $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ are made (in contrast to the linear equation of the theoretical model)_
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
# </details>

# In[ ]:


# They are derived by first retriveing the coefficients (which are 
# stored in fitted_model.params). Then we calculate the fitted values by
# using the equation Yi^ = B0^ + B1^Xi. After, we implement
#fitted_model.fittedvalues.


# ### 5. Explain concisely in your own words what line is chosen for the fitted model based on observed data using the "ordinary least squares" method (as is done by *trendline='ols'* and *smf.ols(...).fit()*) and why it requires "squares"<br>
#     
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _This question addresses the use of **residuals** $\text{e}_i = \hat \epsilon_i = Y_i - \hat y_i$ (in contrast to the **error** terms $\epsilon_i$ of the theoretical model), and particularly, asks for an explanation based on the following visualization_
# >
# > ```python 
# > import scipy.stats as stats
# > import numpy as np
# > import pandas as pd
# > import statsmodels.formula.api as smf
# > import plotly.express as px
# > 
# > n,x_min,x_range,beta0,beta1,sigma = 20,5,5,2,3,5
# > x = stats.uniform(x_min, x_range).rvs(size=n)
# > errors = stats.norm(loc=0, scale=sigma).rvs(size=n)
# > Y = beta0 + beta1 * x + errors
# > 
# > df = pd.DataFrame({'x': x, 'y': Y})
# > model_data_specification = smf.ols("Y~x", data=df) 
# > fitted_model = model_data_specification.fit() 
# > 
# > df['Data'] = 'Data' # hack to add data to legend 
# > fig = px.scatter(df, x='x',  y='Y', color='Data', 
# >                  trendline='ols', title='Y vs. x')
# > 
# > # This is what `trendline='ols'` is
# > fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
# >                 line=dict(color='blue'), name="trendline='ols'")
# > 
# > x_range = np.array([df['x'].min(), df['x'].max()])
# > y_line = beta0 + beta1 * x_range
# > fig.add_scatter(x=x_range, y=y_line, mode='lines',
# >                 name=str(beta0)+' + '+str(beta1)+' * x', 
# >                 line=dict(dash='dot', color='orange'))
# > 
# > # Add vertical lines for residuals
# > for i in range(len(df)):
# >     fig.add_scatter(x=[df['x'][i], df['x'][i]],
# >                     y=[fitted_model.fittedvalues[i], df['Y'][i]],
# >                     mode='lines',
# >                     line=dict(color='red', dash='dash'),
# >                     showlegend=False)
# >     
# > # Add horizontal line at y-bar
# > fig.add_scatter(x=x_range, y=[df['Y'].mean()]*2, mode='lines',
# >                 line=dict(color='black', dash='dot'), name='y-bar')
# > 
# > fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
# > ```
# >
# > _**Question Scope Warning**: we are not looking for any explanation realted to the mathematical equations for the line chosen for the **Simple Linear Regression** model by the "ordinary least squares" method, which happen to be_
# > 
# > _$$\hat \beta_1 = r_{xy}\frac{s_y}{s_x} \quad \text{ and } \quad  \hat\beta_0 = \bar {y}-\hat \beta_1\bar {x}$$_
# >
# > _where $r_{xy}$ is the **correlation** between $x$ and $Y$ and $s_x$ and $s_Y$ are the **sample standard deviations** of $x$ and $y$_
# >
# > ---
# > 
# > ```python 
# > # Use this if you need it    
# > # https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
# > import plotly.offline as pyo
# > # Set notebook mode to work in offline
# > pyo.init_notebook_mode()    
# > ```
# >
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
#     
# </details>

# In[ ]:


# OLS finds the line that minimzes: 
# SSE (Sum of Squared Errors) = ∑.i(Yi - yi^)^2
# This apporach ensures a best fit by reducing the impact of outliers 
# and emphasizing larger errors, leading to a more reliable model.


# ### 6. Explain why the first expression below can be interpreted as "the proportion of variation in (outcome) Y explained by the model (i.e. _fitted_model.fittedvalues_)"; and therefore, why _fitted_model.rsquared_ can be interpreted as a measure of the accuracy of the model; and, therefore what the two _np.corrcoef(...)[0,1]\*\*2_ expressions capture in the context of _Simple Linear Regression models_.
# 
# 1. `1-((Y-fitted_model.fittedvalues)**2).sum()/((Y-Y.mean())**2).sum()`
# 2. `fitted_model.rsquared`
# 3. `np.corrcoef(Y,fitted_model.fittedvalues)[0,1]**2`
# 4. `np.corrcoef(Y,x)[0,1]**2`<br><br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _**R-squared** is the "the proportion of variation in (outcome) $Y$ explained by the model ($\hat y_i$)" and is defined as_
# >
# > _$R^2 = 1 - \frac{\sum_{i=1}^n(Y_i-\hat y)^2}{\sum_{i=1}^n(Y_i-\bar Y)^2}$_
# >
# > _The visuzation provided in the previous problem can be used to consider $(Y_i-\bar Y)^2$ as the squared distance of the $Y_i$ to their sample average $\bar Y$ as opposed to the squared **residuals** $(Y_i-\hat y)^2$ which is the squared distance of the $Y_i$ to their fitted (predicted) values $Y_i$._
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
# </details>

# In[ ]:


# Proportion of Variation Explained: The expression 1 - RSS/TSS and R^2 
# measure the proportion of variation in Y explained by the model.

# Model Accuracy: fitted_model.rsquared is an indicator of model 
# accuracy, with higher values indicating a better fit.

# Squared Correlation Coefficient: Both 
# np.corrcoef(Y, fitted_values)[0,1]**2 and np.corrcoef(Y, X)[0,1]**2
# yield R^2 in a simple linear regresoin, reflecting the strength of the 
# linear relationship between X and Y.


# ### 7. Indicate a couple of the assumptions of the *Simple Linear Regression* model specification that do not seem compatible with the example data below<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _Hint: What even ARE the assumptions of the  **Simple Linear Regression** model, you ask? Have a look at the mathematical specification and see if what it seems to be assuming._
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
# </details>

# In[ ]:


import pandas as pd
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots

# This data shows the relationship between the amount of fertilizer used and crop yield
data = {'Amount of Fertilizer (kg) (x)': [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 
                                          2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 
                                          4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 
                                          6.4, 6.6, 6.8, 7, 7.2, 7.4, 7.6, 7.8, 8, 
                                          8.2, 8.4, 8.6, 8.8,9, 9.2, 9.4, 9.6],
        'Crop Yield (tons) (Y)': [18.7, 16.9, 16.1, 13.4, 48.4, 51.9, 31.8, 51.3, 
                                  63.9, 50.6, 58.7, 82.4, 66.7, 81.2, 96.5, 112.2, 
                                  132.5, 119.8, 127.7, 136.3, 148.5, 169.4, 177.9, 
                                  186.7, 198.1, 215.7, 230.7, 250.4, 258. , 267.8, 
                                  320.4, 302. , 307.2, 331.5, 375.3, 403.4, 393.5,
                                  434.9, 431.9, 451.1, 491.2, 546.8, 546.4, 558.9]}
df = pd.DataFrame(data)
fig1 = px.scatter(df, x='Amount of Fertilizer (kg) (x)', y='Crop Yield (tons) (Y)',
                  trendline='ols', title='Crop Yield vs. Amount of Fertilizer')

# Perform linear regression using scipy.stats
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(df['Amount of Fertilizer (kg) (x)'], df['Crop Yield (tons) (Y)'])
# Predict the values and calculate residuals
y_hat = intercept + slope * df['Amount of Fertilizer (kg) (x)']
residuals = df['Crop Yield (tons) (Y)'] - y_hat
df['Residuals'] = residuals
fig2 = px.histogram(df, x='Residuals', nbins=10, title='Histogram of Residuals',
                    labels={'Residuals': 'Residuals'})

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Crop Yield vs. Amount of Fertilizer', 
                                    'Histogram of Residuals'))
for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)
fig.update_layout(title='Scatter Plot and Histogram of Residuals',
    xaxis_title='Amount of Fertilizer (kg)', yaxis_title='Crop Yield (tons)',
    xaxis2_title='Residuals', yaxis2_title='Frequency', showlegend=False)

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[4]:


# Linearity Assumption:
# - Assumption: SLR assumes a linear relationship between the predictor
# X (Amount of Fertilizer) and the outcome Y (Crop Yield)

# - Observation: The data shows a trend where crop yeild increases at a
# slow rate with low fertilizer ammounts, then accelerates with modern 
# fertilizer use, and eventually increases more rapidly at higher fertilizer
# levels. This suggests a non-linear relationship between fertilizer and
# crop yeild, which a simple linea regreassion model would struggle to 
# capture accurately. The patter indicates that a polynomial or non-linear
# regression model might better fit the data


# Normality of Residuals:
# - Assumption: For valid inferance, SLR assumes that the residuals
# (differences between observed and predicted values) are normally 
# distributed

# - Observation: If you observe the histogram of residuals (differences
# it's likely that they are not normally distributed, which suggests 
# that a linear model may not be capturing the true relationship in 
# the data. For example, if the residuals appear skewed or show any 
# systematic pattern (2.g. fanning out or clustering), this would 
# violate the normality assumption, impacting the reliability of 
# standard errors and confidence intervals


# ## "Week of Nov04" HW [due prior to the Nov08 TUT]
# 
# _**In place of the "Data Analysis Assignment" format we introduced for the previous weeks' HW, the remaining questions will be a collection of exercises based around the following data**_
# 
# > The details of the "LOWESS Trendline" shown below are not a part of the intended scope of the activities here, but it is included since it is suggestive of the questions we will consider and address here
# 

# In[ ]:


import plotly.express as px
import seaborn as sns
import statsmodels.api as sm

# The "Classic" Old Faithful Geyser dataset: ask a ChatBot for more details if desired
old_faithful = sns.load_dataset('geyser')

# Create a scatter plot with a Simple Linear Regression trendline
fig = px.scatter(old_faithful, x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions", 
                 trendline='ols')#'lowess'

# Add a smoothed LOWESS Trendline to the scatter plot
lowess = sm.nonparametric.lowess  # Adjust 'frac' to change "smoothness bandwidth"
smoothed = lowess(old_faithful['duration'], old_faithful['waiting'], frac=0.25)  
smoothed_df = pd.DataFrame(smoothed, columns=['waiting', 'smoothed_duration'])
fig.add_scatter(x=smoothed_df['waiting'], y=smoothed_df['smoothed_duration'], 
                mode='lines', name='LOWESS Trendline')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# ### 8. Specify a *null hypothesis* of "no linear association (on average)" in terms of the relevant *parameter* of the *Simple Linear Regression* model, and use the code below to characterize the evidence in the data relative to the *null hypothesis* and interpret your subsequent beliefs regarding the Old Faithful Geyser dataset.<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _Remember that **Hypothesis Testing** is not a "mathematical proof"_
# >
# > - _We do not prove $H_0$ false, we instead give evidence against the $H_0$: "We reject the null hypothesis with a p-value of XYZ, meaning we have ABC evidence against the null hypothesis"_
# > - _We do not prove $H_0$ is true, we instead do not have evidence to reject $H_0$: "We fail to reject the null hypothesis with a p-value of XYZ"_
# 
# |p-value|Evidence|
# |-|-|
# |$$p > 0.1$$|No evidence against the null hypothesis|
# |$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
# |$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
# |$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
# |$$0.001 \ge p$$|Very strong evidence against the null hypothesis|
# 
# </details>    
# 
# > ```python
# > import seaborn as sns
# > import statsmodels.formula.api as smf
# >
# > # The "Classic" Old Faithful Geyser dataset
# > old_faithful = sns.load_dataset('geyser')
# > 
# > linear_for_specification = 'duration ~ waiting'
# > model = smf.ols(linear_for_specification, data=old_faithful)
# > fitted_model = model.fit()
# > fitted_model.summary()
# > ```
# 

# In[6]:


# In the context of a Simple Linear Regression, we can specify the null
# hypothesis as: H0: B1 = 0, where B1 is the slope of the regression line.
# Under this hypothesis, the average change in eruption duration is not
# associated with changes in waiting time, implying a lack of linear relationship

# Interpretations of Results: 
# 1) Slope Coefficient: The slope coefficient (from mode1.params) tells
# us the estimated change in eruption duration for each additional unit
# increases in waiting time.

# 2) P-Value: The p-value associated with the slope parameter 
# (in mode1.summary()) is the probability of observing such a relationship
# (or one more extreme) between waiting time and duration under the null 
# hypothesis. A low p-value (typically < 0.05) suggets that we can reject 
# the null hypothesis, providing evidence for a linear relationship between
# waiting time and duration

# 3) Subsequent Beliefs: If the p-value is small and we reject H0, this 
# indicates that waiting time likely has a significant, linear association
# with eruption duration, supporting a linear model. However, the LOWESS
# trendline (in the scatter plot) may provide additional insight if the 
# relationship is non-linear, indicating that the linear model might not
# fully capture the pattern, and a non-linear model could provide a better
# fit.


# ### 9. As seen in the introductory figure above, if the delay of the geyser eruption since the previous geyser eruption exceeds approximately 63 minutes, there is a notable increase in the duration of the geyser eruption itself. In the figure below we therefore restrict the dataset to only short wait times. Within the context of only short wait times, is there evidence in the data for a relationship between duration and wait time in the same manner as in the full data set? Using the following code, characterize the evidence against the *null hypothesis* in the context of short wait times which are less than  *short_wait_limit* values of *62*, *64*, *66*.<br>
# 

# In[ ]:


# By examining the slop, p-values, and R^2 values for each subset of
# short wait times, we can determine whether the relationship observed in 
# the full dataset holds within shorter waiting periods. This anaylysis
# helps clarify whether the notable relationship seen in the dataset is
# primarily due to longer waiting times or if a significant association
# exists even when wait times are restricted


# In[ ]:


import plotly.express as px
import statsmodels.formula.api as smf


short_wait_limit = 62 # 64 # 66 #
short_wait = old_faithful.waiting < short_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[short_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (<"+str(short_wait_limit)+")", 
                 trendline='ols')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# ### 10. Let's now consider just the (*n=160*) long wait times (as specified in the code below), and write code to do the following:
# 
# 1. create fitted **Simple Linear Regression** models for **boostrap samples** and collect and visualize the **bootstrapped sampling distribution** of the **fitted slope coefficients** of the fitted models;  
# 
# 
# 2. **simulate** samples (of size `n=160`) from a **Simple Linear Regression** model that uses $\beta_0 = 1.65$, $\beta_1 = 0$, $\sigma = 0.37$ along with the values of `waiting` for $x$ to create **simuations** of $Y$ and use these collect and visualize the **sampling distribution** of the **fitted slope coefficient** under a **null hypothesis** assumption of "no linear association (on average)"; then,  
# 
# 
# 3. report if $0$ is contained within a 95\% **bootstrapped confidence interval**; and if the **simulated p-value** matches `smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1]`?<br><br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _You'll need to create `for` loops to repeatedly create fitted **Simple Linear Regression** models using different samples, collecting the **fitted slope coeffient** created in each `for` loop "step" in order to visualize the **simulated sampling distributions**_
# > 
# > - _A **bootstrapped sample** of the "long wait times" dataset can be created with `old_faithful[long_wait].sample(n=long_wait.sum(), replace=True)`_
# >
# >
# > - _A **simulated** version of the "long wait times under a null hypothesis assumption of **no linear association (on average)**" dataset can be created by first creating `old_faithful_simulation = old_faithful[long_wait].copy()` and then assigning the **simulated** it values with `old_faithful_simulation['duration'] = 1.65 + 0*old_faithful_simulation.waiting + stats.norm(loc=0, scale=0.37).rvs(size=long_wait.sum())`_ 
# >
# >  _The values $\beta_0 = 1.65$ and $\sigma = 0.37$ are chosen to match what is actually observed in the data, while $\beta_1 = 0$ is chosen to reflect a **null hypothesis** assumption of "no linear assocaition (on average)"; and, make sure that you understand why it is that_
# >
# >
# > - _if `bootstrapped_slope_coefficients` is the `np.array` of your **bootstrapped slope coefficients** then `np.quantile(bootstrapped_slope_coefficients, [0.025, 0.975])` is a 95\% **bootstrapped confidence interval**_
# > 
# >
# > - _if `simulated_slope_coefficients` is the `np.array` of your **fitted slope coefficients** **simulated** under a **null hypothesis** "no linear association (on average)" then `(np.abs(simulated_slope_coefficients) >= smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().params[1]).mean()` is the **p-value** for the **simulated** **simulated sampling distribution of the slope coeficients** under a **null hypothesis** "no linear association (on average)"_
# 
# </details>
# <br>

# In[ ]:


import plotly.express as px

long_wait_limit = 71
long_wait = old_faithful.waiting > long_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[long_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (>"+str(long_wait_limit)+")", 
                 trendline='ols')
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[ ]:


# Bootstraped Slope Confidence Interval: This interval represents the 
# range of slope values we would expect by resampling from the observed 
# data. If the interval includes zero, it suggets weak evidence for 
# a strong positive or negative relationship within the long wait times

# Null Hypothesis Distribution and Simulated p-value: By simulating data
# with B1 = 0, we create a distribution of slope estimates expected under
# no association. The proportion of null slope estimates greater in 
# magnitude than the observed slope gives a simulated p-value, which tells
# us the probability of observing our data if the true slope were actually
# zero

# Code Below:
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
from scipy import stats

# Set a threshold for long wait times
long_wait_limit = 71
long_wait = old_faithful.waiting > long_wait_limit

# Extract data for long waits
long_wait_data = old_faithful[long_wait]

# Step 1: Bootstrapped Sampling Distribution of Slope Coefficients
n_bootstraps = 1000
boot_slope_coefs = []

for _ in range(n_bootstraps):
    # Bootstrap sample
    sample = long_wait_data.sample(frac=1, replace=True)
    # Fit linear regression model and store the slope
    model = smf.ols('duration ~ waiting', data=sample).fit()
    boot_slope_coefs.append(model.params['waiting'])

# Visualize bootstrapped slope distribution
fig = px.histogram(boot_slope_coefs, nbins=30, title="Bootstrapped Sampling Distribution of Slope Coefficients")
fig.show(renderer="png")

# Step 2: Simulate under Null Hypothesis (beta1 = 0)
null_slope_coefs = []
beta0, beta1, sigma = 1.65, 0, 0.37

for _ in range(n_bootstraps):
    # Generate simulated Y values using null hypothesis parameters
    y_simulated = beta0 + beta1 * long_wait_data['waiting'] + np.random.normal(0, sigma, len(long_wait_data))
    # Fit linear regression model on simulated data
    sim_data = pd.DataFrame({'waiting': long_wait_data['waiting'], 'duration': y_simulated})
    model_null = smf.ols('duration ~ waiting', data=sim_data).fit()
    null_slope_coefs.append(model_null.params['waiting'])

# Visualize null hypothesis slope distribution
fig = px.histogram(null_slope_coefs, nbins=30, title="Sampling Distribution of Slope Coefficients Under Null Hypothesis")
fig.show(renderer="png")

# Step 3: Calculate 95% Confidence Interval and Simulated P-Value
# Bootstrapped Confidence Interval for observed slopes
lower_bound = np.percentile(boot_slope_coefs, 2.5)
upper_bound = np.percentile(boot_slope_coefs, 97.5)

# Check if 0 is within this interval
contains_zero = lower_bound <= 0 <= upper_bound

# Original linear regression model on long wait data
original_model = smf.ols('duration ~ waiting', data=long_wait_data).fit()

# Calculated simulated p-value
p_value_sim = (np.abs(null_slope_coefs) >= np.abs(original_model.params['waiting'])).mean()

# Report findings
print(f"95% Bootstrapped Confidence Interval for slope: [{lower_bound:.3f}, {upper_bound:.3f}]")
print("Contains zero:", contains_zero)
print(f"Simulated p-value for slope coefficient under null hypothesis: {p_value_sim:.3f}")
print("Model Summary Slope P-Value:", original_model.summary().tables[1].as_html())


# ### 11. Since we've considered wait times of around <64  "short" and wait times of >71 "long", let's instead just divide the data and insead call wait times of <68 "short" and otherwise just call them "long". Consider the *Simple Linear Regression* model specification using an *indicator variable* of the wait time length<br>
# 
# $$\large Y_i = \beta_{\text{intercept}} + 1_{[\text{"long"}]}(\text{k_i})\beta_{\text{contrast}} + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$
# 
# ### where we use $k_i$ (rather than $x_i$) (to refer to the "kind" or "katagory" or "kontrast") column (that you may have noticed was already a part) of the original dataset; and, explain the "big picture" differences between this model specification and the previously considered model specifications<br>
# 
# 1. `smf.ols('duration ~ waiting', data=old_faithful)`
# 2. `smf.ols('duration ~ waiting', data=old_faithful[short_wait])`
# 3. `smf.ols('duration ~ waiting', data=old_faithful[long_wait])`
# 
# ### and report the evidence against a *null hypothesis* of "no difference between groups "on average") for the new *indicator variable* based model<br>
# 

# In[ ]:


from IPython.display import display

display(smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit().summary().tables[1])

fig = px.box(old_faithful, x='kind', y='duration', 
             title='duration ~ kind',
             category_orders={'kind': ['short', 'long']})
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[ ]:


# 1. Model Summary Table:
# - Specifically, it will include a row labeled C(kind, Treatment(
# reference="short"))[T.long], representing the difference in eruption 
# duration between “long” and “short” wait times.)
# - The coefficient for this row shows the estimated increase in 
# eruption duration for “long” wait times compared to “short” wait 
# times.
# - The p-value for this coefficient tells us whether this difference 
# is statistically significant.

# 2. Box Plot:
# - The box plot visually represents the distribution of eruption 
# durations for “short” and “long” wait times.
# - If “long” wait times consistently lead to longer durations, 
# we should see the “long” group centered higher on the y-axis than 
# the “short” group.

# Interpretations: 
# If the coefficient for C(kind, Treatment(reference="short"))[T.long] 
# is significantly different from zero (as indicated by a small p-value,
# typically  p < 0.05 ), this suggests strong evidence that “long” wait
# times are associated with longer eruption durations on average 
# compared to “short” wait times. The box plot will provide additional
# visual support, showing whether the median and overall range of 
# eruption durations for “long” wait times are consistently greater 
# than for “short” wait times.


# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
# </details>

# ### 12. Identify which of the histograms suggests the plausibility of the assumption that the distribution of *error* terms is normal for each of the models, and explain why the other three do not support this assumption.
# 
# > Hint: Question 5 of the *Communication Activity #2* of the Oct25 TUT (addressing an *omitted* section of the TUT) discusses how the assumption in *Simple Linear Regression* that the *error* terms $\epsilon_i \sim \mathcal N\left(0, \sigma\right)$ is diagnostically assessed by evaluating distributional shape of the *residuals* $\text{e}_i = \hat \epsilon_i = Y_i - \hat y_i$
# 

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats
import numpy as np

model_residuals = {
    '<br>Model 1:<br>All Data using slope': smf.ols('duration ~ waiting', data=old_faithful).fit().resid,
    '<br>Model 2:<br>Short Wait Data': smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().resid,
    '<br>Model 3:<br>Long Wait Data': smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().resid,
    '<br>Model 4:<br>All Data using indicator': smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit().resid
}

fig = make_subplots(rows=2, cols=2, subplot_titles=list(model_residuals.keys()))
for i, (title, resid) in enumerate(model_residuals.items()):

    if i == 1:  # Apply different bins only to the second histogram (index 1)
        bin_size = dict(start=-1.9, end=1.9, size=0.2)
    else:
        bin_size = dict(start=-1.95, end=1.95, size=0.3)

    fig.add_trace(go.Histogram(x=resid, name=title, xbins=bin_size, histnorm='probability density'), 
                  row=int(i/2)+1, col=(i%2)+1)
    fig.update_xaxes(title_text="n="+str(len(resid)), row=int(i/2)+1, col=(i%2)+1)    
    
    normal_range = np.arange(-3*resid.std(),3*resid.std(),0.01)
    fig.add_trace(go.Scatter(x=normal_range, mode='lines', opacity=0.5,
                             y=stats.norm(loc=0, scale=resid.std()).pdf(normal_range),
                             line=dict(color='black', dash='dot', width=2),
                             name='Normal Distribution<br>(99.7% of its area)'), 
                  row=int(i/2)+1, col=(i%2)+1)
    
fig.update_layout(title_text='Histograms of Residuals from Different Models')
fig.update_xaxes(range=[-2,2])
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[8]:


# Model 1: All Data Using Slope:
# - If this histogram shows a roughly symmetric and bell-shaped 
# distribution with most residuals concentrated near zero and tapering 
# off similarly on both sides, it suggests normality. This distribution
# would indicate that the normality assumption is plausible for the 
# residuals when using all data.

# Model 2: Short Wait Data:
# - If the histogram of residuals for short wait times is skewed,
# has outliers, or deviates significantly from a bell shape, it would 
# indicate that the normality assumption is less plausible. Since this 
# subset focuses only on short wait times, it’s possible that 
# residuals may not capture the full distributional pattern, leading to
# deviation from normality.

# Model 3: Long Wait Data: 
# - Similar to the short wait data, if the histogram here diverges from
# a normal distribution by showing skewness, heavy tails, or other 
# irregular shapes, this suggests a poor fit to the normality
# assumption. This may occur if “long” wait times contain systematic
# variations or outliers that skew the distribution

# Model 4: All Data Using Indicator Variable:
# - If this histogram appears approximately normal, it suggests that 
# using an indicator variable for “short” and “long” wait times results
# in residuals that align well with the normality assumption. However,
# if the residuals are non-symmetric or show pronounced peaks or gaps, 
# it would indicate non-normality.


# ### 13. The "short" and "long" wait times are not "before and after" measurements so there are not natural pairs on which to base differences on which to do a "one sample" (paired differences) *hypothesis test*; but, we can do "two sample" hypothesis testing using a *permuation test*, or create a 95% *bootstrap confidence interval* for the difference in means of the two populations. 
# 
# ### (A) Do a permuation test $\;H_0: \mu_{\text{short}}=\mu_{\text{long}} \; \text{ no difference in duration between short and long groups}$ by "shuffling" the labels
# ### (B) Create a 95% bootstrap confidence interval  by repeatedly bootstrapping within each group and applying *np.quantile(bootstrapped_mean_differences, [0.025, 0.975])* to the collection of differences between the sample means.    
# ### (a) Explain how the sampling approaches work for the two simulations.
# ### (b) Compare and contrast these two methods with the *indicator variable* based model approach used in Question 11, explaining how they're similar and different.<br>
#     
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _You'll need to create `for` loops for repeated (shuffling simulation) **permutation** and (subgroup) **bootstrapping**, where_
# >
# > - _"shuffling" for **permutation testing** is done like this `old_faithful.assign(kind_shuffled=old_faithful['kind'].sample(n=len(old_faithful), replace=False).values)#.groupby('kind').size()`; then, the **mean difference statistic** is then calculated using `.groupby('kind_shuffled')['duration'].mean().iloc[::-1].diff().values[1]` (so the **observed statistic** is `old_faithful.groupby('kind')['duration'].mean().iloc[::-1].diff().values[1]`_
# > 
# >
# > - _"two sample" **bootstrapping** is done like this `old_faithful.groupby('kind').apply(lambda x: x.sample(n=len(x), replace=True)).reset_index(drop=True)#.groupby('kind').size()`; then, the **bootstrapped mean difference statistic** is then calculated using `.groupby('kind')['duration'].mean().iloc[::-1].diff().values[1]` (like the **observed statistic** except this is applied to the **bootstrapped** resampling of `old_faithful`)_
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_
# </details>

# In[13]:


# A) 
# 1) Observed Difference: First, compute the observed difference in 
# means between the "short" and "long" groups
# 2) Shuffling Labels: Shuffle the labels between the two groups many times
# (e.g. 10,000 iterations) and recompute the dufference in means each time
# 3) Compare: The proportion of permuted differences that are as extreme 
# as or more extreme than the observed difference provides the p-value.
# If this p-value is small (e.g. < 0.05), reject the null hypothesis.

# This test provides a non-parametric way to asses whether there is a
# significant difference between the groups without making assumptions 
# about the distribution of the data

# B)
# 1) Resample: Resample the “short” and “long” groups with replacement
# to create bootstrap samples.
# 2) Compute Difference: Calculate the difference in means between the
# “short” and “long” groups for each bootstrap sample.
# 3) Repeat: Repeat this process many times (e.g., 10,000 iterations).
# 4) Confidence Interval: The 2.5th and 97.5th percentiles of the 
# bootstrapped differences form the 95% confidence interval.

# This method estimates the uncertainty around the difference in means by
# simulating sampling from the population and does not require the assumption
# of normality

# a) 
# 1) Permutation Test: The permutation test tests the null hypothesis by 
# randomizing the group labels and assessing how extreme the observed 
# difference in means is under the null hypothesis.
# 2) Bootstrap Confidence Interval: In the bootstrap method, we resample 
# with replacement from each group to simulate what the sampling 
# distribution of the difference in means might look like under 
# repeated sampling from the same population.

# Both methods are resampling techniques that do not assume a specific
# distribution for the data and provide robust ways to assess the 
# difference between the groups.

# b) 
# Similarity: All three approaches aim to assess the difference in 
# durations between the “short” and “long” groups. However, they 
# differ in how they approach the problem:
# 1) The indicator variable-based model in regression 
# (from Question 11) uses a categorical predictor to model the 
# difference in means between the two groups. It estimates the 
# difference as part of the regression model and tests its 
# significance using standard hypothesis testing (e.g., t-tests for 
# coefficients).
# 2) The permutation test and bootstrap confidence interval are 
# non-parametric methods that do not rely on any assumptions about the 
# distribution of the data. They instead resample the data to assess 
# the difference between the groups.
# Difference: 
# 1) The indicator variable-based model makes assumptions about the
# linear relationship between the indicator variable (“kind”) and the
# response variable, and it uses regression to estimate and test the
# difference in means.
# 2) The permutation test and bootstrap confidence interval offer 
# more flexible, distribution-free alternatives by resampling the data,
# making them robust to non-normal distributions and other assumptions
# required by parametric methods.

# Conclusion: 
# - Indicator Variable-Based Model: Provides a formal, parametric 
# approach using linear regression to test for differences between 
# groups.
# - Permutation Test: A non-parametric, resampling-based test that 
# evaluates the observed difference by comparing it to the distribution
# of differences generated by random label swapping.
# - Bootstrap Confidence Interval: Another non-parametric approach that
# estimates the uncertainty around the difference in means by 
# resampling with replacement.


# In summary, the indicator-varaible based model assumes a parametric 
# relationship and tests the difference via model coefficients, while the
# permutations and bootstrap methods do not rely on distributional 
# assumptions and instead resaple the data to assess the difference 
# between groups.


# ### 14. Have you reviewed the course wiki-textbook and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# >  _Here is the link of [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) in case it gets lost among all the information you need to keep track of_  : )
# >
# > _Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)_
# </details>    

# In[ ]:


# yesssirrrrr


# ## Recommended Additional Useful Activities [Optional]
# 
# The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course 
# 
# <br>
# <details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>
# 
# ### Ethical Professionalism Considerations
#     
# The TUT and HW both addressed some of the assumptions used in **Simple Linear Regression**. The **p-values** provided by `statsmodels` via `smf.ols(...).fit()` depend on these assumptions, so if they are not (at least approximately) correct, the **p-values** (and any subsequent claims regarding the "evidience against" the **null hypothesis**) are not reliable. In light of this consideration, describe how you could diagnostically check the first three assumptions (given below) when using analyses based on **Simple Linear regression** model. From an Ethical and Professional perspective, do you think doing diagnostic checks on the assumptions of a **Simple Linear regression** model is something you can and should do whenever you're doing this kind of analysis? 
#             
# > The first three assumptions associated with the **Simple Linear regression** model are that
# > 
# > - the $\epsilon_i$ **errors** (sometimes referred to as the **noise**) are **normally distributed**
# > - the $\epsilon_i$ **errors** are **homoscedastic** (so their distributional variance $\sigma^2$ does not change as a function of $x_i$)
# > - the linear form is [at least reasonably approximately] "true" (in the sense that the above two remain [at least reasonably approximately] "true") so that then behavior of the $Y_i$ **outcomes** are represented/determined on average by the **linear equation**)<br>
# > 
# >    and there are additional assumptions; but, a deeper reflection on these is "beyond the scope" of STA130; nonetheless, they are that<br><br>
# > - the $x_i$ **predictor variable** is **measured without error**
# > - and the $\epsilon_i$ **errors** are **statistically independent** (so their values do not depend on each other)
# > - and the $\epsilon_i$ **errors** are **unbiased** relative to the **expected value** of **outcome** $E[Y_i|x_i]=\beta_0 + \beta_1x_i$ (which is equivalently stated by saying that the mean of the **error distribution** is $0$, or again equivalently, that the **expected value** of the **errors** $E[\epsilon_i] = 0$)
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
# > ### NEW DEVELOPMENT<br>New Abilities Achieved and New Levels Unlocked!!!    
# > **As noted, the Week 01 HW introduced the [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb) notebook.** _And there it instructed students to explore the notebook through the first 16 cells of the notebook._ The following cell in that notebook (there marked as "run cell 17") is preceded by an introductory section titled, "**Now for some comparisons...**", _**and all material from that point on provides an example to allow you to start applying what you're learning about Hypothesis Testing to the CSCS data**_ **using a paired samples ("one sample") framework.**
# >
# > **NOW, HOWEVER, YOU CAN DO MORE.** 
# > - _**Now you can do "two sample" hypothesis testing without the need for paired samples.**_ All you need are two groups.
# > - _**And now you can do simple linear regression modeling.**_ All you need are two columns.
# 
# ### Current Course Project Capability Level
# 
# At this point in the course you should be able to do a **Simple Linear Regression** analysis for data from the Canadian Social Connection Survey data
#     
# 1. Create and test a **null hypothesis** of no linear association "on average" for a couple of columns of interest in the Canadian Social Connection Survey data using **Simple Linear Regression**
# 
# 2. Use the **residuals** of a fitted **Simple Linear Regression** model to diagnostically assess some of the assumptions of the analysis
# 
# 3. Use an **indicator variable** based **Simple Linear Regression** model to compare two groups from the Canadian Social Connection Survey data
# 
# 4. Compare and contrast the results of an **indicator variable** based **Simple Linear Regression** model to analyses based on a **permutation test** and a **bootstrapped confidence interval**   
#     
# </details>    

# In[ ]:





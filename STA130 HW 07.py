#!/usr/bin/env python
# coding: utf-8

# # STA130 Homework 07
# 
# Please see the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) for the list of topics covered in this homework assignment, and a list of topics that might appear during ChatBot conversations which are "out of scope" for the purposes of this homework assignment (and hence can be safely ignored if encountered)
# 

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

# 
# ### Marking Rubric (which may award partial credit) 
# 
# - [0.1 points]: All relevant ChatBot summaries [including link(s) to chat log histories if you're using ChatGPT] are reported within the notebook
# - [0.3 points]: Well-communicated, clear demonstration of the "model building" process and techniques of "Question 4"
# - [0.3 points]: Well-communicated, clear demonstration of the "model building" process and techniques of "Question 7"
# - [0.3 points]: Well-communicated, clear demonstration of the "model building" process and techniques of "Question 9"
# 

# ## "Pre-lecture" HW [*completion prior to next LEC is suggested but not mandatory*]
# 

# ### 1. Explain succinctly in your own words (but working with a ChatBot if needed)...<br>
# 
# 1. the difference between **Simple Linear Regression** and **Multiple Linear Regression**; and the benefit the latter provides over the former
# 
# 
# 2. the difference between using a **continuous variable** and an **indicator variable** in **Simple Linear Regression**; and these two **linear forms**
# 
# 
# 3. the change that happens in the behavior of the model (i.e., the expected nature of the data it models) when a single **indicator variable** is introduced alongside a **continuous variable** to create a **Multiple Linear Regression**; and these two **linear forms** (i.e., the **Simple Linear Regression** versus the **Multiple Linear Regression**)
# 
# 
# 4. the effect of adding an **interaction** between a **continuous** and an **indicator variable** in **Multiple Linear Regression** models; and this **linear form**
# 
# 
# 5. the behavior of a **Multiple Linear Regression** model (i.e., the expected nature of the data it models) based only on **indicator variables** derived from a **non-binary categorical variable**; this **linear form**; and the necessarily resulting **binary variable encodings** it utilizes
#        
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _The requested **linear forms** are **equations**, and answers must include **equations** along with the explanations that interpret the **linear forms**. Write you **linear forms** using a style something like_
# > 
# > - _"outcome = $\beta_0$ + $\beta_A$ predictorA + $\beta_B$ 1(predictorB)"_ 
# > - _where the "1(.)" notation is for indicator variables_ 
# > - _or feel free to use an similar alternative if a ChatBot provides you with another notation you think is clearer and like better if you prefer_
# >
# > _DO INCLUDE the **intercept** in your **linear forms**. You don't have to include notation related to the **error** term since this is essentially always assumed (and, actually, we usually don't even bother to include the **intercept** in such shorthand specifications either, for the same reason), but don't forget to include the **intercept** here this time (for practice). The modeling **assumptions** do not need to be addressed beyond this, but explanations will likely address the number of variables and the essential use-case (perhaps illustrated through examples) the different models imply._    
# > 
# > _Answers to the final question above should address the notion of a "baseline" group and it's role for **model interpretation**, why "number of categories minus one" **indicator variables** are used to represent the original **categorical variable**, and the relationship between the **binary** and **categorical variables** that are relevant for this model specification. An example use-case would likely be helpful for illustration here._ 
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details> 

# In[3]:


#1. Simple Linear Regression(SLR) uses only one predictor variable to model
# the relationship with the response variable, while Multiple Linear
# Regression(MLR) incorporates two or more predictors. The benefit of MLR is 
# its ability to capture more complexity in relationships as it can
# account for the combined influrence of multiple predictors.

#2. A continous variable in SLR allows the model to capture relationships
# along a continous scale, like height or age. An indicator variable on
# the other hand, represents categorical information with binary values
# (e.g. 0 or 1 for gender). These lead to two types of linear forms in
# SLR: one with a slope reflecting the continous change and another with a 
# shift in intercept for categorical differences.

#3. Adding an indicator variable alongside a continous predictor in MLR 
# allows the model to capture both the effect of the continous variable
# and any categorical shifts associated with the indicator. This form
# differentiates between levels of the categorical variable while also 
# modelling the continous variable's influence.

#4. An interaction term between a continous and an indicator variable in
# MLR allows the model to capture different slopes for the continous 
# predictor across levels of the categorical variable. This form reflects
# situations where the effect of the continous variable varies depending
# on the category.

#5. When using only indicator variables derived from a non-binary 
# categorical variable, MLR encodes each category (beyond a reference
# group) as a seperate binary variable. This leads to a piecewise model
# where each binary variable shifts the intercepts accoridng to its
# category, capturing the effect of categorical differences without
# continous variation in predictors.


# ### 2. Explain in your own words (but working with a ChatBot if needed) what the specific (outcome and predictor) variables are for the scenario below; whether or not any meaningful interactions might need to be taken into account when predicting the outcome; and provide the linear forms with and without the potential interactions that might need to be considered<br>
# 
# > Imagine a company that sells sports equipment. The company runs advertising campaigns on TV and online platforms. The effectiveness of the TV ad might depend on the amount spent on online advertising and vice versa, leading to an interaction effect between the two advertising mediums.    
# 
# 1. Explain how to use these two formulas to make **predictions** of the **outcome**, and give a high level explaination in general terms of the difference between **predictions** from the models with and without the **interaction** 
# 
# 2. Explain how to update and use the implied two formulas to make predictions of the outcome if, rather than considering two continuous predictor variables, we instead suppose the advertisement budgets are simply categorized as either "high" or "low" (binary variables)    
#     
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _While working on this question, it's important to clearly understand the (**outcome** and **predictor**) **variables** under consideration, and they way they are being considered. Similarly to the previous (first) question of this homework assignment, this question requires the **equations** of the indicated **linear forms** and an explanation of their **interpretation** and use. What is different here is that the **interactions** being considered are between two **continuous variables** or two **binary variables** (for a total of four **equations** under consideration where two include the **interactions** and two do not)._
# >
# > _The way an **interaction** actually appears and works in the context of the linear form of a **multiple linear regression** model may not necessarily be immediately intuitive, as it is actually in fact somewhat subtle and tricky. Of course, an **interaction** is when the relationship of one **predictor variable** with the **outcome variable** depends on the value of another different **predictor variable**, so the impact of one **predictor variable** changes based on the presence or magnitude of another **predictor variable**. But are you sure you know what this means in the context of the **linear form** of a **multiple linear regression** model?_
# >
# > - _Imagine you're selling smoothies, the taste of your smoothie depends on the ingredients included in the smoothie, and there are two ingredients (bananas and strawberries) in the smoothie._
# >     - _Adding more bananas into the smoothie will of course increase the "banana" flavor of the smoothie, and vice-versa for strawberries..._
# >     - _But is this "banana" influence on the flavor always constant, or can it change depending on the absolute amount of strawberries in the smoothie?_ 
# >     - _If the "banana" flavor influence is constant and does not depend on the  absolute amount of strawberries in the smoothie, then there is no **interaction** and the **linear form** of the model is $\beta_b b_i + \beta_s s_i$ and the model is said to be only **additive**._
# >     - _But if the "banana" flavor influence does depend on the absolute amount of strawberries in the smoothie, then there IS an **interaction** and the **linear form** of the model is $\beta_b b_i + \beta_s s_i + \beta_{bs} (b_i\times s_i)$ and the model is said to be **synergistic**._
# >         
# > _These **linear forms** show that either bananas and strawberries do not have any **synergistic interaction** and contribute to the flavor independently; or, they do have a **synergistic interaction** and there is an interesting interplay between bananas and strawberries in the way they influence the taste of the smoothie._ 
# > 
# > - _So, if there is no **interaction**, then the effect of adding more bananas on the taste of the smoothie will always be the same, no matter how many strawberries you put in. So the effect of bananas on the smoothie is the same whether you add a lot of strawberries or just a few: $\beta_b b_i + \beta_s s_i$_
# > - _Or, on the other hand, if there is an **interaction**, then the effect of adding bananas (on the smootie flavor) will be different depending on how many strawberries there currently are in the smoothie: $\beta_b b_i + \beta_s s_i + \beta_{bs} (b_i\times s_i)$_
# > 
# > _In this case, the right answer is probably that the **linear form** with the **interaction** is correct. This is because the flavor probably depends on the relative amount of bananas and strawberries in the smoothie; so, the effect of adding a fixed amount of bananas to the smoothie probalby depends on the absolute amount of strawberries that are in the smoothie._
# > 
# > _Again, because understanding **interactions** in the context of **linear forms** is somewhat subtle and tricky and indeed not necessarily obviously intuitive, let's think about this a bit more. And we can simplify the concept a little bit by considering how this **interaction** would actually technically work in a **linear form** if we just had **binary indicator variables**._
# >         
# > - _To consider the smootie example in terms of binary variables, suppose that if both fruits are added to the smootie, they will be added in the same amount. So the smoothie will be made with either just bananas, just strawberries, or both (or neither and you won't make a smoothie)._ 
# >     - _The question regarding an **interaction** then is, is the influence of the ingredients on the taste of the smoothie **additive** or **synergistic**? That is, does the way bananas affects the flavor of the smoothie change depending on the inclusion or exclusion of strawberries in the smoothie?_
# >     - _**Additive** $\beta_b 1_{[b_i=1]}(b_i) + \beta_s 1_{[s_i=1]}(s_i)$ means there are three different flavors but they are explained by just two **parameters**: banana $\beta_b$, strawberry $\beta_s$, and banana-strawberry $\beta_b+\beta_s$_
# >     - _**Synergistic** $\beta_b 1_{[b_i=1]}(b_i) + \beta_s 1_{[s_i=1]}(s_i) + \beta_{bs}(1_{[b_i=1]}(b_i) \times 1_{[s_i=1]}(s_i))$ means there are of course again three different flavors, but this time they are explained by three **parameters**: banana $\beta_b$, strawberry $\beta_s$, and banana-strawberry $\beta_b+\beta_s + \beta_{bs}$, which indicates that the flavor is "more than just sum of its parts", meaning there is a **synergistic interaction** and there is an interesting interplay between bananas and strawberries in the way they influence the taste of the smoothie_
# >     
# > _As the **additive** and **synergistic** versions of the **linear form** of the two **binary indicator variables** context shows, we don't need an interaction to make different predictions for different combinations of things. Instead, what these show is that the prediction will either be **additive** and "just the sum of it's parts" or **synergistic** (**interactive**) and "more than just sum of its parts"._
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#       
# </details>
# 

# In[ ]:


# 1. Simple Linear Regression (SLR) uses one predictor variable, while 
# Multiple Linear Regression (MLR) uses multiple predictors, caputuring
# more complex relationships. MLR provides a benefit by accounting for
# the combined influence of multiple variables, potentially improving 
# prediction accuracy. 

# 2. A continous variable allows the model to show gradual changes in 
# the outcome, while an indicator variable represents categories with 
# binary values (0 or 1). These two forms lead to different linear 
# models: one with a continous slope, and another with shifts in 
# intercepts to capture categorical effects.


# ### 3. Use *smf* to fit *multiple linear regression* models to the course project dataset from the canadian social connection survey<br>
# 
# > **EDIT: No, you probably actually care about CATEGORICAL or BINARY outcomes rather than CONTINUOUS outcomes... so you'll probably not actually want to do _multiple linear regression_ and instead do _logistic regression_ or _multi-class classification_. Okay, I'll INSTEAD guide you through doing _logistic regression_.**
# 
# 1. ~~for an **additive** specification for the **linear form** based on any combination of a couple **continuous**, **binary**, and/or **categorical variables** and a **CONTINUOUS OUTCOME varaible**~~ 
#     1. This would have been easy to do following the instructions [here](https://www.statsmodels.org/dev/example_formulas.html). A good alternative analagous presentation for logistic regression I just found seems to be this one from a guy named [Andrew](https://www.andrewvillazon.com/logistic-regression-python-statsmodels/). He walks you through the `logit` alternative to `OLS` given [here](https://www.statsmodels.org/dev/api.html#discrete-and-count-models).
#     2. Logistic is for a **binary outcome** so go see this [piazza post](https://piazza.com/class/m0584bs9t4thi/post/346_f1) describing how you can turn any **non-binary categorical variable** into a **binary variable**. 
#     3. Then instead do this problem like this: **catogorical outcome** turned into a **binary outcome** for **logistic regression** and then use any **additive** combination of a couple of **continuous**, **binary**, and/or **categorical variables** as **predictor variables**. 
# 
# 
# ```python
# # Here's an example of how you can do this
# import pandas as pd
# import statsmodels.formula.api as smf
# 
# url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
# pokeaman = pd.read_csv(url).fillna('None')
# 
# pokeaman['str8fyre'] = (pokeaman['Type 1']=='Fire').astype(int)
# linear_model_specification_formula = \
# 'str8fyre ~ Attack*Legendary + Defense*I(Q("Type 2")=="None") + C(Generation)'
# log_reg_fit = smf.logit(linear_model_specification_formula, data=pokeaman).fit()
# log_reg_fit.summary()
# ```
# 
# 
# 2. ~~for a **synertistic interaction** specification for the **linear form** based on any combination of a couple **continuous**, **binary**, and/or **categorical variables**~~
#     1. But go ahead and AGAIN do this for **logistic regression** like above.
#     2. Things are going to be A LOT simpler if you restrict yourself to **continuous** and/or **binary predictor variables**.  But of course you could *use the same trick again* to treat any **categorical variable** as just a **binary variable** (in the manner of [that piazza post](https://piazza.com/class/m0584bs9t4thi/post/346_f1).
#     
# 
# 3. and **interpretively explain** your **linear forms** and how to use them to make **predictions**
#     1. Look, intereting **logistic regression** *IS NOT* as simple as interpreting **multivariate linear regression**. This is because it requires you to understand so-called **log odds** and that's a bit tricky. 
#     2. So, INSTEAD, **just intepret you logistic regression models** *AS IF* they were **multivariate linear regression model predictions**, okay?
# 
# 
# 4. and interpret the statistical evidence associated with the **predictor variables** for each of your model specifications 
#     1. **Yeah, you're going to be able to do this based on the `.fit().summary()` table _just like with multiple linear regression_**... now you might be starting to see how AWESOME all of this stuff we're doing is going to be able to get...
# 
# 
# 5. and finally use `plotly` to visualize the data with corresponding "best fit lines" for a model with **continuous** plus **binary indicator** specification under both (a) **additive** and (b) **synergistic** specifications of the **linear form** (on separate figures), commenting on the apparent necessity (or lack thereof) of the **interaction** term for the data in question
#     1. Aw, shit, you DEF not going to be able to do this if you're doing **logistic regression** because of that **log odds** thing I mentioned... hmm...
#     2. OKAY! Just *pretend* it's **multivariate linear regression** (even if you're doing **logistic regression**) and *pretend* your **fitted coefficients** belong to a **continuous** and a **binary predictor variable**; then, draw the lines as requested, and simulate **random noise** for the values of your **predictor data** and plot your lines along with that data.
#     
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _This [link](https://www.statsmodels.org/dev/examples/notebooks/generated/formulas.html) offers guidance on using `statsmodels.formula.api` (`smf`) to build statistical models in Python using formulas._
# >
# > _The "best fit lines" summarize the relationship between the **outcome** and **predictor variables** observed in the data as well as the **linear form** of the **multiple linear regression** allows. The statistical evidence for the these estimated realtionship characterizations of course depends on an evaluation of the **hypothesis testing** for the **coefficients** of the model. **Model building** is the process of exploring the evidence for observed relationships captured through the modeling of the data in order to arrive at reliable (**generalizable**) claims based on the data, and perhaps make predictions about the future based on these created beliefs and understandings (whose value of course depends on how trustworthy these created beliefs and understandings are)._
# >
# > _When we do not find sufficient sufficient evidence for supposed relationships that we'd like to leverage for understanding or prediction, attempting to move forward on the basis of such "findings" is certainly a dangerous errand..._
#     
# </details>
# 

# In[1]:


# 1. Select a categorical outcome variable and convert it to a binary 
# form. For instance, if you have a categorical variable with several
# levels (e.g. social engagement level: low, medium, high), convert it 
# to a binary form (e.g. "high engagement" vs "not high engangement").

# 2. Using smf.logit from statsmodels, set up an additive model 
# combining two predictors (continous, binary, or a mix) For example,
# if you're interested in age (continous) and employment status (binary),
# your model might look like:
log_reg_formula = 'binary_outcome ~ age + employment_status'
log_reg_model = smf.logit(log_reg_formula, data=data).fit()
# Check log_reg_model.summary() to review the statistical significance
# of ecah predictor.

# 3. To explore potential interactions, add an interaction term to the
# formula:
log_reg_formula_interaction = 'binary_outcome ~ age * employment_status'
log_reg_model_interaction = smf.logit(log_reg_formula_interaction, data=data).fit()
# interpret whether the interaction term significantly affects the
# outcome by checking its p-value.

# 4. Simulate a continous predictor, a binary predictor, and random 
# noise to generate data resembling a continous outcome.
# For both the additive and interaction models, use plotly to plot "best
# fit lines" by pretending the logistic regression coefficients are for
# linear regression. This will allow a visual comparison of the additive
# and synergistic effects.

# 5. Use the .summary() output to interpret the influence of each 
# predictor and the interaction term, if included. Describe the 
# significance of each term as if they contribute linearly to the 
# outcome.


# ### 4. Explain the apparent contradiction between the factual statements regarding the fit below that "the model only explains 17.6% of the variability in the data" while at the same time "many of the *coefficients* are larger than 10 while having *strong* or *very strong evidence against* the *null hypothesis* of 'no effect'"<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#     
# > _How do we simultaneously interpret **hypothesis testing** results regarding **coefficient estimates** based on **p-values** and **R-squared** "the proportion of variation in (outcome) $y$ explained by the model ($\hat y_i$)"? How can both be meaningfully understood at the same time? Do they address different aspects of a model?_
# >    
# > _As introduced in the previous homework, **R-squared** is_
# >
# > _$$R^2 = 1 - \frac{\sum_{i=1}^n(y_i-\hat y)^2}{\sum_{i=1}^n(y_i-\bar y)^2}$$_
# >    
# > _which describes the **explanatory power** of a model; whereas, **p-values** allow us to characterize **evidence against** a **null hypothesis**, and **coefficients** in a **multiple linear regression** context allow us to interpret the relationship between the **outcome** and a **predictor variable** "with all other **predictor variables** 'held constant'". Are these concepts thus contradictory or conflictual in some manner?_
# 
# |p-value|Evidence|
# |-|-|
# |$$p > 0.1$$|No evidence against the null hypothesis|
# |$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
# |$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
# |$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
# |$$0.001 \ge p$$|Very strong evidence against the null hypothesis|
#     
# > _In `formula='HP ~ Q("Sp. Def") * C(Generation)'` the `Q` stands for "quote" and is needed to access column names when they have a "space" in their name, while the `C` indicates a **categorical** use of what is actually an **integer** valued column. Despite technically being **continuous** numbers, **integer** often simply indicate categories which should not necessarily be treated as an incremental **continuous predictor variable**. Remember, a model such as $\beta_0 + \beta_1 x$ means for each unit increase in $x$ the outcome increases "on average" by $\beta_1$; so, if $x$ takes on the values `1` through `6` as the `Generation` **predictor variable** here does, then this means the average value for "Generation 1" must be $\beta_0 + \beta_1$ while for "Generation 2" it must be $\beta_0 + 2\times \beta_1$ (and so on up to "Generation 6" which must be $\beta_0 + 6\times \beta_1$). This might be a very strange restriction to place on something that is really actually a **categorical predictor variable**. You can see in the given model fit below how this six-level **categorical predictor variable** is actually appropriately treated in the specification of the **linear form** using "Generation 1" for the "baseline" and **binary indicators** to model the "contrast" ("offsets") for the other "Generations"; and, how these are in turn used in the context of the **interaction** considered by the model specification._ 
#     
# </details>

# In[ ]:


import pandas as pd

url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
# fail https://github.com/KeithGalli/pandas/blob/master/pokemon_data.csv
pokeaman = pd.read_csv(url) 
pokeaman


# In[ ]:


import statsmodels.formula.api as smf

model1_spec = smf.ols(formula='HP ~ Q("Sp. Def") + C(Generation)', data=pokeaman)
model2_spec = smf.ols(formula='HP ~ Q("Sp. Def") + C(Generation) + Q("Sp. Def"):C(Generation)', data=pokeaman)
model2_spec = smf.ols(formula='HP ~ Q("Sp. Def") * C(Generation)', data=pokeaman)

model2_fit = model2_spec.fit()
model2_fit.summary()


# In[3]:


# we have an apparent contradiction where the model explains only 17.6% 
# of the variability in the data (low R^2 value), and many coefficients
# are large (greater than 10) and show strong evidence against the null
# hypothesis (significant p-values). THis situation can occur due to two
# main reasons:

# 1. Low R^2 Value (Explained Variability): The R^2 statistic indicates
# how much of the variability in the dependent variable (here, HP) is 
# explained by the model. A low  R^2  (e.g., 17.6%) suggests that the
# predictors (e.g., Sp. Def and Generation) only explain a small 
# fraction of the variation in HP. In other words, many other factors 
# influence HP that are not captured by this model.

# 2. Signifant Coefficients (Evidence Against Null Hypothesis): Although
# the overall fit of the model is weak (as seen in the low  R^2 ), 
# individual coefficients can still have strong evidence against the 
# null hypothesis, showing that they are statistically significant.


# <details class="details-example"><summary style="color:blue"><u>Continue now...?</u></summary>
# 
# ### Pre-lecture VS Post-lecture HW
#     
# Feel free to work on the "Post-lecture" HW below if you're making good progress and want to continue: in this case the "Post-lecture" HW just builds on the "Post-lecture" HW, introducing and extending the considerations available in the **multiple linear regression context**. That said, as "question 3" above hopefully suggests and reminds you, the **course project** is well upon us, and prioritizing work on that (even over the homework) may very well be indicated at this point...
# 
# *The benefits of continue would are that (a) it might be fun to try to tackle the challenge of working through some problems without additional preparation or guidance; and (b) this is a very valable skill to be comfortable with; and (c) it will let you build experience interacting with ChatBots (and beginning to understand their strengths and limitations in this regard)... it's good to have sense of when using a ChatBot is the best way to figure something out, or if another approach (such as course provided resources or a plain old websearch for the right resourse) would be more effective*
#     
# </details>    
# 

# ## "Post-lecture" HW [*submission along with "Pre-lecture" HW is due prior to next TUT*]
# 

# ### 5. Discuss the following (five cells of) code and results with a ChatBot and based on the understanding you arrive at in this conversation explain what the following (five cells of) are illustrating<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#     
# > _Recall from the previous week's homework that the **R-squared** "the proportion of variation in (outcome) $y$ explained by the model ($\hat y_i$)" quantity (re-introduced in the previous problem) can be calculated as `np.corrcoef(y,fitted_model.fittedvalues)[0,1]**2` (as well as several other ways in the **simple linear regression** context). The **squared correlation** between the outcome $y$ and it's **fitted values** $\hat y$ is the most generally useful formulation of **R-squared** since this can be use in the **multiple linear regression** context._
# > 
# > _This question therefore thus addresses the question of model **generalizability** on the basis of "in sample" and "out of sample" **model performance** (measured by **R-squared**)._
# > 
# > - _The **squared correlation** between the **outcomes** $y$ and their **fitted values** $\hat y$ is an "in sample" **model performance** metric since the $\hat y$ "predictions" for the $y$ **outcomes** are based on using those already **observed outcomes** to fit the model to generate the $\hat y$._  
# > 
# > - _If we instead calculate **squared correlation** between **outcomes** $y$ that were not used to fit the model and their corresponding $\hat y$ **predictions** (which are indeed now actually **predictions** as opposed to **fitted values**), then we are now  calculating an "out of sample" **model performance** metric._
# > 
# > _When an "out of sample" metric performs more poorly than a comparitive "in sample" metric, then the **predictions** of the **fitted model** are not **generalizing** to data being the dataset the model is fit on. In this case we say the model is **overfit** (to the data its fit was based on). The purpose of using different **training** and **testing** datasets is to consider "in sample" versus "out of sample" **model performance** in order to try to confirm that the model is not **overfit** and that the **predictions** do indeed seem to **generalizable** beyond the dataset used for **model fitting**._
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_
#     
# </details>

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split

fifty_fifty_split_size = int(pokeaman.shape[0]*0.5)

# Replace "NaN" (in the "Type 2" column with "None")
pokeaman.fillna('None', inplace=True)

np.random.seed(130)
pokeaman_train,pokeaman_test = \
  train_test_split(pokeaman, train_size=fifty_fifty_split_size)
pokeaman_train


# In[ ]:


model_spec3 = smf.ols(formula='HP ~ Attack + Defense', 
                      data=pokeaman_train)
model3_fit = model_spec3.fit()
model3_fit.summary()


# In[ ]:


yhat_model3 = model3_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model3_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model3)[0,1]**2)


# In[ ]:


model4_linear_form = 'HP ~ Attack * Defense * Speed * Legendary'
model4_linear_form += ' * Q("Sp. Def") * Q("Sp. Atk")'
# DO NOT try adding '* C(Generation) * C(Q("Type 1")) * C(Q("Type 2"))'
# That's 6*18*19 = 6*18*19 possible interaction combinations...
# ...a huge number that will blow up your computer

model4_spec = smf.ols(formula=model4_linear_form, data=pokeaman_train)
model4_fit = model4_spec.fit()
model4_fit.summary()


# In[ ]:


yhat_model4 = model4_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model4_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model4)[0,1]**2)


# In[6]:


# These 5 cells illustrate:

#1. Data Splitting and Preparation: The importance of preparing data, 
# including filling missing values and splitting into training and 
# testing sets, to enable a fair assessment of model generalization.

#2. Model Comparison: Model 3 (simple) and Model 4 (complex) show two
# different approaches to regression: Model 3 uses a straightforward
# additive form, Model 4 includes interaction terms to capture potential
# synergistic effects between predictors.

#3. Evaluation of In-Sample vs Out-of-Sample Performance: By comparing
# R^2  values for both in-sample and out-of-sample performance, the 
# cells illustrate the importance of checking if a model generalizes 
# beyond the training data. This helps highlight whether added 
# complexity leads to improved out-of-sample performance 
# (suggesting useful complexity) or overfitting.


# ### 6. Work with a ChatBot to understand how the *model4_linear_form* (*linear form* specification of  *model4*) creates new *predictor variables* as the columns of the so-called "design matrix" *model4_spec.exog* (*model4_spec.exog.shape*) used to predict the *outcome variable*  *model4_spec.endog* and why the so-called *multicollinearity* in this "design matrix" (observed in *np.corrcoef(model4_spec.exog)*) contribues to the lack of "out of sample" *generalization* of *predictions* from *model4_fit*; then, explain this consisely in your own works<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _The **overfitting** observed in the previous problem is a question of **model complexity** relative to the amount of information or evidence available in a given dataset (or we could just say "the amount of data in a dataset"). The **model fit** for  `model4` resulted in an **overfit model** because the specification of its **linear form** was overly complex (relative to the the amount of available data). Indeed, `model4` is rediculously complex as can be seen from `model4_fit.summary()`. This in turn allowed the **model fit** to "detect" idiosyncratic associations spuriously present specifically in the **training** dataset but which did not **generalize** to the **testing** dataset. If a model is too **complex** then it will find and model "patterns" in a **training** dataset which are actually just accidental "noise" from the random sampling nature of the dataset. The simpler (or more parsimoneous) `model3` on the other hand was able to identify **predictive associations** in the **training** dataset which did **generalize** to the **testing** dataset. This is because `model3` only sought to understand simpler **predictive association** for which there was enough data in the **training** dataset to reliably detect and obviously identify. And these patterns were indeed sufficiently "real" in the sense that they were present and **generalized** into the **testing** dataset as well as the **training** dataset. So they could be "found" in the **training** and then used in (**generalized** to) the **testing** dataset._
# > 
# > _This question therefore addresses the topic of the **evidence** a given dataset provides for the **predictive associations** detected by a **fitted model**. It should be increasingly clear at this point that evidence for a model can be addressed using **coefficent hypothesis testing** in the context of **multiple linear regression**, but that examinations of "in sample" versus "out of sample" **model performance** metrics are what in fact are directly designed to address this question of **generalizability**. That said, this question introduces another consideration of **multicollinearity** as something that affects the **generalizability** of **model fits** in the **multiple linear regression** context. A good question that a ChatBot could help you understand is (a) "why is **generalizability** more uncertain if two **predictor variables** are highly **correlated**?" and (b) "why is **generalizability** more uncertain if multiple **predictor variables** are highly **multicollinear**?"_
# >
# > _The four code cells below are not necessary for answering this question; however, they introduce two very practical helpful tools for the **multiple linear regression** context that are immediately relevant for this question. The first is the so-called **condition number** (of a "design matrix") which provides a very simple diagnostic which can serve as a measure the degree of **multicollinearity** that is present in a **model fit**. If this number is "very large" then there is a large degree of **multicollinearity** and suggests room for doubt regarding the **generalizability** of the **fitted model**. The second tool(s) are the `center` and `scale` functions. It is best practice to "center and scale" **continuous predictor variables** (but not **indicator variables**) in the **multiple linear regression** context as is done below. While "centering and scaling" does make interpreting the predictions on the original scale of the data slighly more complicated, it also must be done in order to get a "true" evaluation of the degree of **multicollinearity** present in a **model fit** using the **condition number** of the model ("design matrix"). The examples below show that the **condition number** reported by a **fitted model** are "artificially inflacted" if "centering and scaling" is not used. Specically, they show that the **condition number** of `model3_fit` is really `1.66` (as opposed to the "very large" `343` which is reported without "centering and scaling"); whereas, the **condition number** for `model4_fit` is "very (VERY) large" irrespective of "centering and scaling", showing that the overwheling presense of **multicollinearity** in `model4_fit` is in fact a very real thing.  Indeed, we have already seen that `model4_fit` is grossly **overfit** and does not remotely **generalize** beyond its **training** dataset. Without knowing this, however, the comically large **condition number** for `model4_fit` (after "centering and scaling") makes it abundantly clear that we should have great doubts about the likely **generalizability** of `model4_fit` (even without examining specific aspects of **multicollinearity** directly or examining "in sample" versus "out of sample" **model performance** comparisions)._
# >
# > - _The "specific aspects of **multicollinearity**" reference above refer to understanding and attributing the detrmimental affects of specific **predictor variables** towards **multicollinearity**. This can be done using so-called **variance inflation factors**, but this is beyond the scope of STA130. We should be aware that the presence of excessive **multicollinearity** as indicated by a large **condition number** for a ("centered and scaled") **fitted model** raises grave concerns regarding the potential **generalizability** of the model._
# >
# > _The `np.corrcoef(model4_spec.exog)` examination of the **correlations** of a "design matrix" considered in ths problems prompt is analogous to the examination of the **correlations** present in a dataset that might considered when initially examining the **predictor variables** of a dataset, such as `pokeaman.iloc[:,4:12].corr()`. Indeed, such an examination is often the first step in examining the potential presence of **multicollinearity** among the **predictor variables** of a dataset. However, these are consideration of **pairwise correlation**, whereas **multicollinearity** generalizes this notion to the full collection of **predictor variables** together. A **condition number** for a "centered and scale" version of a **fit model** can therefore be viewed as serving the analogous purposes of a multivariate generalization of **pairwise correlation**._
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details>
# 

# In[ ]:


# "Cond. No." WAS 343.0 WITHOUT to centering and scaling
model3_fit.summary() 


# In[ ]:


from patsy import center, scale

model3_linear_form_center_scale = \
  'HP ~ scale(center(Attack)) + scale(center(Defense))' 
model_spec3_center_scale = smf.ols(formula=model3_linear_form_center_scale,
                                   data=pokeaman_train)
model3_center_scale_fit = model_spec3_center_scale.fit()
model3_center_scale_fit.summary()
# "Cond. No." is NOW 1.66 due to centering and scaling


# In[ ]:


model4_linear_form_CS = 'HP ~ scale(center(Attack)) * scale(center(Defense))'
model4_linear_form_CS += ' * scale(center(Speed)) * Legendary' 
model4_linear_form_CS += ' * scale(center(Q("Sp. Def"))) * scale(center(Q("Sp. Atk")))'
# Legendary is an indicator, so we don't center and scale that

model4_CS_spec = smf.ols(formula=model4_linear_form_CS, data=pokeaman_train)
model4_CS_fit = model4_CS_spec.fit()
model4_CS_fit.summary().tables[-1]  # Cond. No. is 2,250,000,000,000,000

# The condition number is still bad even after centering and scaling


# In[ ]:


# Just as the condition number was very bad to start with
model4_fit.summary().tables[-1]  # Cond. No. is 12,000,000,000,000,000


# In[ ]:


# This code illustrates the impact of centering, scaling, and
# multicollinearity on model stability and out-of-sample generalization
# in a regression analysis.

#1. Centering and Scaling: By centering and scaling predictors 
# (like Attack and Defense), we reduce multicollinearity, lowering the 
# condition number and making coefficients more stable. For a simpler 
# model, this adjustment brought the condition number to 1.66, 
# indicating low multicollinearity.

#2. High Multicollinearity with Interactions: In a more complex model 
# with interaction terms (e.g., Attack * Defense * Speed * Legendary),
# multicollinearity is severe even after centering and scaling, with a 
# condition number over 2 quadrillion. High multicollinearity inflates 
# coefficient variances and makes the model unstable and unreliable for
# new data.

#3. Design Matrix: The interaction terms create highly correlated 
# columns in the design matrix, which worsens multicollinearity, 
# leading to poor out-of-sample performance, as seen in the lower 
# out-of-sample  R^2  values.


# ### 7. Discuss with a ChatBot the rationale and principles by which *model5_linear_form* is  extended and developed from *model3_fit* and *model4_fit*; *model6_linear_form* is  extended and developed from *model5_linear_form*; and *model7_linear_form* is  extended and developed from *model6_linear_form*; then, explain this breifly and consisely in your own words<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _We again include the **condition number** for the "centered and scaled" version of `model7_fit` to show that **multicollinearity** does not appear to be a major concern for this model (and the same would be true regarding `model6_fit` if the analogous "centered and scaled" version of the model was considered). While it is true that the **condition number** of `15.4` observed for `model7_fit` is perhaps "large", this would not be considered "vary large"._
# >
# > - _Regarding **condition numbers**, a ChatBot gave me cutoffs of `<30` not a big problem, up to `<300` maybe an issue, up to `<1000` definitely **multicollinearity**, and beyond that is pretty much likely to be "serious" problems with **multicollinearity**. Personally, cutoffs around `10`, `100`, and `1000` seem about right to me._
# >
# > _This question addresses the **model building** exercise using both an **evidence** based approach using **coefficient hypothesis testing** as well as examinations of **generalizability** using comparisions of "in sample" versus "out of sample" **model performance** metrics. Through these tools, different models were considered, extended, and developed, finally arriving at `model7_fit`. When we feel we can improve the **model performance** in a **generalizable** manner, then all relatively underperforming models are said to be **underfit**, meaning that they do not leverage all the **predictive associations** available to improve **predictions**._
# > 
# > _While the previous "Question 6" above introduced and explored the impact of **multicollinearity** in the **multiple linear regression** context_ 
# >     
# > - _(whereby "the effects" of multiple **predictor variables** are "tangled up" and therefore do not allow the model to reliably determine contribution attributions between the **predictor variables**, which potentially leads to poor **estimation** of their "effects" in the model, which in turn is the problematic state of affairs which leads to a lack of **generalizability** in such high **multicollinearity** settings)_
# > 
# > _there is still the (actually even more important) consideration of the actual **evidence** of **predictive associations**. The question is whether or not there is sufficient **evidence** in the data backing up the **estimated** fit of the **linear form** specification. Quantifying the **evidence** for a **estimated** model is a separate question from the problem of **multicollinearity**, the assessment of which is actually the primary purpose of **multiple linear regression** methodology._
# >    
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details>    

# In[ ]:


# Here's something a little more reasonable...
model5_linear_form = 'HP ~ Attack + Defense + Speed + Legendary'
model5_linear_form += ' + Q("Sp. Def") + Q("Sp. Atk")'
model5_linear_form += ' + C(Generation) + C(Q("Type 1")) + C(Q("Type 2"))'

model5_spec = smf.ols(formula=model5_linear_form, data=pokeaman_train)
model5_fit = model5_spec.fit()
model5_fit.summary()


# In[ ]:


yhat_model5 = model5_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model5_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model5)[0,1]**2)


# In[ ]:


# Here's something a little more reasonable...
model6_linear_form = 'HP ~ Attack + Speed + Q("Sp. Def") + Q("Sp. Atk")'
# And here we'll add the significant indicators from the previous model
# https://chatgpt.com/share/81ab88df-4f07-49f9-a44a-de0cfd89c67c
model6_linear_form += ' + I(Q("Type 1")=="Normal")'
model6_linear_form += ' + I(Q("Type 1")=="Water")'
model6_linear_form += ' + I(Generation==2)'
model6_linear_form += ' + I(Generation==5)'

model6_spec = smf.ols(formula=model6_linear_form, data=pokeaman_train)
model6_fit = model6_spec.fit()
model6_fit.summary()


# In[ ]:


yhat_model6 = model6_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model6_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model6)[0,1]**2)


# In[ ]:


# And here's a slight change that seems to perhaps improve prediction...
model7_linear_form = 'HP ~ Attack * Speed * Q("Sp. Def") * Q("Sp. Atk")'
model7_linear_form += ' + I(Q("Type 1")=="Normal")'
model7_linear_form += ' + I(Q("Type 1")=="Water")'
model7_linear_form += ' + I(Generation==2)'
model7_linear_form += ' + I(Generation==5)'

model7_spec = smf.ols(formula=model7_linear_form, data=pokeaman_train)
model7_fit = model7_spec.fit()
model7_fit.summary()


# In[ ]:


yhat_model7 = model7_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model7_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model7)[0,1]**2)


# In[ ]:


# And here's a slight change that seems to perhas improve prediction...
model7_linear_form_CS = 'HP ~ scale(center(Attack)) * scale(center(Speed))'
model7_linear_form_CS += ' * scale(center(Q("Sp. Def"))) * scale(center(Q("Sp. Atk")))'
# We DO NOT center and scale indicator variables
model7_linear_form_CS += ' + I(Q("Type 1")=="Normal")'
model7_linear_form_CS += ' + I(Q("Type 1")=="Water")'
model7_linear_form_CS += ' + I(Generation==2)'
model7_linear_form_CS += ' + I(Generation==5)'

model7_CS_spec = smf.ols(formula=model7_linear_form_CS, data=pokeaman_train)
model7_CS_fit = model7_CS_spec.fit()
model7_CS_fit.summary().tables[-1] 
# "Cond. No." is NOW 15.4 due to centering and scaling


# In[ ]:


# "Cond. No." WAS 2,340,000,000 WITHOUT to centering and scaling
model7_fit.summary().tables[-1]


# In[ ]:


# This progression of models demonstrates how incremental adjustments 
# to a regression model's specficiation can lead to potential improvements
# in fit and predictive power, but also introduce complexity and
# multicollinearity.

#1. Model5: Extends previous models by adding continuous predictors 
# (Sp. Def and Sp. Atk) and categorical variables like Type 1, Type 2,
# and Generation. This increases the model’s ability to capture more 
# nuanced effects, but also introduces more parameters and interactions.

#2. Model6: Builds on Model5 by incorporating interaction terms and 
# creating binary indicators for specific types (Type 1 = Normal and 
# Type 1 = Water) and generations (Generation = 2 and Generation = 5). 
# This allows the model to differentiate effects based on these 
# specific categories, potentially improving predictive power.

#3. Model7: Adds further interactions to capture any synergistic
# effects between the existing predictors. However, the complexity 
# continues to grow, increasing the risk of overfitting and 
# multicollinearity.

#4. Model7 (Centered and Scaled): Centers and scales continuous 
# predictors to manage multicollinearity, significantly reducing the 
# condition number. This helps mitigate instability in the coefficient
# estimates but indicates that high multicollinearity persists due to 
# categorical interactions.


# ### 8. Work with a ChatBot to write a *for* loop to create, collect, and visualize many different paired "in sample" and "out of sample" *model performance* metric actualizations (by not using *np.random.seed(130)* within each loop iteration); and explain in your own words the meaning of your results and purpose of this demonstration<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#     
# > _The following code could likely be slightly edited and repurposed to match the 50-50 **train-test split** analysis and data above (in the `train_test_split` method of "Question 5").  Considering the behavior of the `model3_fit` **linear form** specification would be the suggested way to start, but it might also be interesting and/or helpful to consider the different available **linear form** specifications in the manner of this problem..._
# >    
# > ```python
# > import plotly.express as px  # etc.
# >
# > songs_training_data,songs_testing_data = train_test_split(songs, train_size=31)
# > linear_form = 'danceability ~ energy * loudness + energy * mode'
# >    
# > reps = 100
# > in_sample_Rsquared = np.array([0.0]*reps)
# > out_of_sample_Rsquared = np.array([0.0]*reps)
# > for i in range(reps):
# >     songs_training_data,songs_testing_data = \
# >       train_test_split(songs, train_size=31)
# >     final_model_fit = smf.ols(formula=linear_form, 
# >                               data=songs_training_data).fit()
# >     in_sample_Rsquared[i] = final_model_fit.rsquared
# >     out_of_sample_Rsquared[i] = \
# >       np.corrcoef(songs_testing_data.danceability, 
# >                   final_model_fit.predict(songs_testing_data))[0,1]**2
# >     
# > df = pd.DataFrame({"In Sample Performance (Rsquared)": in_sample_Rsquared,
# >                    "Out of Sample Performance (Rsquared)": out_of_sample_Rsquared})   >  
# > fig = px.scatter(df, x="In Sample Performance (Rsquared)", 
# >                      y="Out of Sample Performance (Rsquared)")
# > fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="y=x", line_shape='linear'))  
# > ```
# >
# > _When you first look at this question, you might be unsure about the specific issue that the code is addressing. Take a moment to think about why the code repeatedly randomly re-splits the data, fits the model, and compares the "in sample" versus "out of sample" **R-squared** values (over and over). Of course, if a **fit model** performs well on the **training** dataset but doesn't do as well on the **testing** dataset then we might be observing the affects of **overfitting**. But why might it sometimes be the opposite situation (which we actually encountered right away for `model3_fit` when the **train-test split** was based on  `np.random.seed(130)` and resulted in a better "out of sample" **R-squared** of about `0.21` vereses the 'in-sample" **R-squared** of about `0.15`)? If you're thinking that this should therefore vice-versa intuitively mean **underfitting**, actually that's not right because **underfitting** is when the **generalizability** of a different model **linear form** specification that provides improved **model performance** is **validated**. What were seeing here, the variable, is something else..._
# >        
# > ---
# > 
# > _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
#     
# </details>
#     

# In[7]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Placeholder for storing "in sample" and "out of sample" R-squared metrics
in_sample_r2s = []
out_of_sample_r2s = []

# Loop to generate multiple paired R-squared values
for i in range(100):
    # Split the data randomly without setting a random seed within each iteration
    pokeaman_train, pokeaman_test = train_test_split(pokeaman, train_size=0.5)
    
    # Define and fit the model on the training data
    model_spec = smf.ols(formula='HP ~ Attack + Defense + Speed + Q("Sp. Def") * Q("Sp. Atk") + Legendary', 
                         data=pokeaman_train)
    model_fit = model_spec.fit()
    
    # Calculate "in sample" R-squared
    in_sample_r2 = model_fit.rsquared
    
    # Predict and calculate "out of sample" R-squared on the test data
    yhat_test = model_fit.predict(pokeaman_test)
    y_test = pokeaman_test.HP
    out_of_sample_r2 = np.corrcoef(y_test, yhat_test)[0, 1] ** 2

    # Append the results to the lists
    in_sample_r2s.append(in_sample_r2)
    out_of_sample_r2s.append(out_of_sample_r2)

# Visualize the distribution of "in sample" and "out of sample" R-squared values
plt.figure(figsize=(12, 6))

# Histogram for "in sample" R-squared values
plt.hist(in_sample_r2s, bins=15, alpha=0.5, label="In Sample R-squared", color="blue")

# Histogram for "out of sample" R-squared values
plt.hist(out_of_sample_r2s, bins=15, alpha=0.5, label="Out of Sample R-squared", color="orange")

plt.xlabel("R-squared")
plt.ylabel("Frequency")
plt.title("Distribution of 'In Sample' vs 'Out of Sample' R-squared Values")
plt.legend()
plt.show()

#1. Purpose of the Loop: The loop randomly splits the dataset, fits the
# model on one half (training data), and evaluates it on the other half
# (test data). This process is repeated multiple times to observe how 
# the model’s performance varies across different data splits.

#2. Collecting Metrics: Each iteration collects the “in sample” 
# (training data) and “out of sample” (test data) R-squared values. 
# The “in sample” R-squared reflects how well the model fits the data 
# it was trained on, while the “out of sample” R-squared reflects its 
# ability to generalize to new data.

#3. Visualization: By plotting the distribution of these paired 
# R-squared values, we can see the range of possible model performances
# due to different train-test splits. The variation in “out of sample”
# R-squared values, especially compared to “in sample” R-squared, 
# illustrates the consistency (or inconsistency) of the model’s 
# predictive power across different splits.

# In Sample vs. Out of Sample: We typically expect the “in sample” 
# R-squared values to be higher on average than “out of sample” values,
# since the model is evaluated on the data it was trained on. However, 
# if “out of sample” R-squared values show a large range or are 
# consistently much lower, it may indicate that the model is 
# overfitting to the training data and does not generalize well.

# Purpose of the Demonstration: This demonstration highlights the 
# importance of considering out-of-sample performance when evaluating 
# models. It also shows how model performance can vary due to 
# randomness in train-test splits, providing insights into the model’s
# robustness and helping identify issues like overfitting or high 
# variance in predictions.


# ### 9. Work with a ChatBot to understand the meaning of the illustration below; and, explain this in your own words<br>
# 
# <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
# > _While we had seemed to **validate** the **generalizability** of `model7_fit` in **model building** exercise of the previous "Question 7" above, as well as the improved **model performance** of `model7_fit` comapred to `model6_fit`, the `model7_fit` model was always nonetheless more complex than `model6_fit` model (as seen by comparing their `.summary()` methods). This complexity, despite the minimal concerns regarding **multicollinearity**, should always have suggested some room for caution. This is because, as previously discussed in "Question 6" above, a complex **linear form** specification can allow a "**model fit** to 'detect' idiosyncratic associations spuriously present specifically in the **training** dataset but which did not **generalize** to the **testing** dataset." Indeed, a close look at the **p-values** in `model7_fit.summary()` will show that the **evidence** (in the data) for many of the **estimated coefficients** of `model7_fit` is in fact not very strong. In comparision, the **evidence** (in the data) for many of the **estimated coefficients** of `model6_fit.summary()` is consistently stronger._
# >
# > _As discussed towards the end of the commentary in the previous "Question 7" above, the primary purpose of **multiple linear regression** methodology is to allow us to assess the **evidence** (in the data) for a given **linear form** specification based on **coefficient hypothesis testing**. In this regard, then, `model6_fit` might be preferred over `model7_fit` despite the better "out of sample" **model performance** of `model7_fit` over `model6_fit`. This may not be enough to convince everyone however, so an additional consideration that might be made here is that the more simpler (more parsimoneous) nature of `model6_fit` should be preferred over `model7_fit` from the perspective of **model interpretability**. Indeed, it is quite unclear how exactly one should think about and understand a four-way **interaction** variable such as `Attack:Speed:Q("Sp. Def"):Q("Sp. Atk")` in conjunction with the whole host of the additional lower order interations. From a **model interpretability** perspective, understanding the meaning of the complex specification of `model7_fit` is "challenging" and "complicated" to say the least._
# >
# > - _There are also often circumstances where **model interpretability** can be MORE IMPORTANT than raw **model performance** in "out of sample" **prediction**._
# > - _This is ESPECIALLY true if **predictive model performance** is relatively comparable between models of two different complexity levels. In such cases, the benefits of better **model interpretability** might provide a clear argument for the simpler (more parsimoneous) model, not to mention the additional potential benefit of more consistent improved **generalizability** over the the more complex model this might offer._
# >
# > _This question drives home the point that a simpler (more parsimoneous) model always offers the potential benefit of more consistent **generalizability**, not to mention **interpretability**, over more complex models. We should *ONLY* use increasingly complex models that without questions outperfrm simler models. The code below illustrates this by further additionally raising the consideration that the random **train-test** approach used above is actually not the most natural one available for our dataset, which has different "Generations". In fact, if we were actually using this model to make **predictions**, we would increasingly acquire more data over time which we would use to make **precictions** about future data which we haven't yet seen, which is what the code demonstrates. And low and behold, this exposes **generalizability** concerns that we missed when we used the dataset in an idealized way and not actually how we would use such a dataset in practice in the real world (where data would arrive sequentially, and current data is used to predict future data). These **generalizability** concerns do affect both models, but the appear to be more problematic for `model7_fit` than `model6_fit`, which is certainly a result of the increased complexity of `model7_fit` which always opens up the possibility of model **overfitting**._
# 
# <details>    

# In[ ]:


model7_gen1_predict_future = smf.ols(formula=model7_linear_form,
                                   data=pokeaman[pokeaman.Generation==1])
model7_gen1_predict_future_fit = model7_gen1_predict_future.fit()
print("'In sample' R-squared:    ", model7_fit.rsquared, "(original)")
y = pokeaman_test.HP
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model7)[0,1]**2, "(original)")
print("'In sample' R-squared:    ", model7_gen1_predict_future_fit.rsquared, "(gen1_predict_future)")
y = pokeaman[pokeaman.Generation!=1].HP
yhat = model7_gen1_predict_future_fit.predict(pokeaman[pokeaman.Generation!=1])
print("'Out of sample' R-squared:", np.corrcoef(y,yhat)[0,1]**2, "(gen1_predict_future)")


# In[ ]:


model7_gen1to5_predict_future = smf.ols(formula=model7_linear_form,
                                   data=pokeaman[pokeaman.Generation!=6])
model7_gen1to5_predict_future_fit = model7_gen1to5_predict_future.fit()
print("'In sample' R-squared:    ", model7_fit.rsquared, "(original)")
y = pokeaman_test.HP
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model7)[0,1]**2, "(original)")
print("'In sample' R-squared:    ", model7_gen1to5_predict_future_fit.rsquared, "(gen1to5_predict_future)")
y = pokeaman[pokeaman.Generation==6].HP
yhat = model7_gen1to5_predict_future_fit.predict(pokeaman[pokeaman.Generation==6])
print("'Out of sample' R-squared:", np.corrcoef(y,yhat)[0,1]**2, "(gen1to5_predict_future)")


# In[ ]:


model6_gen1_predict_future = smf.ols(formula=model6_linear_form,
                                   data=pokeaman[pokeaman.Generation==1])
model6_gen1_predict_future_fit = model6_gen1_predict_future.fit()
print("'In sample' R-squared:    ", model6_fit.rsquared, "(original)")
y = pokeaman_test.HP
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model6)[0,1]**2, "(original)")
print("'In sample' R-squared:    ", model6_gen1_predict_future_fit.rsquared, "(gen1_predict_future)")
y = pokeaman[pokeaman.Generation!=1].HP
yhat = model6_gen1_predict_future_fit.predict(pokeaman[pokeaman.Generation!=1])
print("'Out of sample' R-squared:", np.corrcoef(y,yhat)[0,1]**2, "(gen1_predict_future)")


# In[ ]:


model6_gen1to5_predict_future = smf.ols(formula=model6_linear_form,
                                   data=pokeaman[pokeaman.Generation!=6])
model6_gen1to5_predict_future_fit = model6_gen1to5_predict_future.fit()
print("'In sample' R-squared:    ", model6_fit.rsquared, "(original)")
y = pokeaman_test.HP
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model6)[0,1]**2, "(original)")
print("'In sample' R-squared:    ", model6_gen1to5_predict_future_fit.rsquared, "(gen1to5_predict_future)")
y = pokeaman[pokeaman.Generation==6].HP
yhat = model6_gen1to5_predict_future_fit.predict(pokeaman[pokeaman.Generation==6])
print("'Out of sample' R-squared:", np.corrcoef(y,yhat)[0,1]**2, "(gen1to5_predict_future)")


# In[1]:


# For model 7 (generation 1): In-sample R-squared (original): The
# R-squared value for the model when it is tested on the data it was
# trained on (generation 1). Out-of-sample R-squared (original): The 
# R-squared value for the model when it is tested on the unseen test 
# data (pokeaman_test).

# For model 7 (excluding Generation 1): In-sample R-squared 
# (gen1_predict_future): The R-squared value when the model is fitted on
# generation 1 data but is used to predict other generations (excluding
# generation 1). Out-of-sample R-squared (gen1_predict__future): The 
# R-squared value when the model is used to predict data from generations
# other than generation 1.

# For model 7 (excluding Generation 6): In-sample R-squared
# (gen1to5_predcit_future): The R-squared value when the model is trained
# using generations 1 to 5 and then used to predict generation 6 data.
# Out-of-sample R-squared (gen1to5_predict_future): The R-squared 
# value when the model is used to predict data from generation 6.

# For model 6 (generation 1): In-sample R-squared (original): The 
# R-squared value for the model when tested on data it was trained on 
# (generation 1). Out-of-sample R-squared (original): The R-squared 
# value for the model when tested on the unseen test data (pokeaman_test).

# For model 6 (excluding Generation 1): In-sample R-squared
# (gen1_predict_future): The R-squared value when the model is trained on
# generation 1 and tested on the remaining generations. Out-of-sample 
# R-squared (gen1_predict_future): The R-squared value for the same but
# predicting future generations.

# For model 6 (excluding generation 6): In-sample R-squared 
# (gen1to5_predict_future): The R-squared value when the model is trained
# on generations 1 to 5 and then used to predict generation 6 data.
# Out-of-sample R-squared (gen1to5_predict_future): The R-squared value 
# when the model is used to predict generation 6 data.


# ## Recommended Additional Useful Activities [Optional]
# 
# The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course
# 
# <br>
# <details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>
# 
# ### Ethical Professionalism Considerations
# 
# This week addresses **multiple linear regression**, perhaps best exemplified through the consideration on **interactions** and their impact on the **model interpretation**, **evidence** and **validity* of models using **coefficient hypothesis testing** and "in sample" versus "out of sample" **model performance** comparision. Exactly, as in **simple linear regression**, the correctness of **p-values** used to give **evidence** for **predictive associations** that are **estimated** from a dataset depends on the (at least approximate) "truth" of the assumptions of the **multiple linear regression**, which are the same as those of the **simple linear regression** with the exception that the specification **linear form** can now model a much richer set of relationships between **predictor** and **outcome variables** based on **predictive associations** observed and **evidenced** in the data. With all this in mind, and reflecting back on the **Ethical Professionalism Considerations** from the previous week concerning **simple linear regression**...
# 
# > - Which of the methods used for diagnostically assessing the assumptions of a **simple linear regression** specification could be used analogously generalized to the **multiple linear regression** context for the same purpose? 
# > 
# > - Examining the assumption of the **linear form** is more challenging in **multiple linear context**, but can be done using so-called **partial regression** (or **added variable**) **plot**. Is a ChatBot able to provide code to perform this diagnostic and instructions regarding its purpose, interpretation, and appropriate usage?
# >     
# > - Are there other diagnostic analyses that a ChatBot might suggest for you to help you evaluate the appropriateness of the assumptions of **fitted multiple linear regression model** you are considering using for **interpretation** or **prediction**? And if so, s the ChatBot able to provide code to perform these additional diagnostic and instructions regarding their purpose, interpretation, and appropriate usages?
# >     
# > - What do you think your ethical and professional responsibilites are when it comes to using and leveraging **multiple linear regression** methodology (and associated assumptions therein) in your work? To illustrate and demonstrate your thoughts on these considerations, can you give any specific examples of decisions that might be made during your process of executing a **multiple linear regression** that could have ethical and professional implications, risks, or consequences? What do you think are the simplest steps can you take to ensure that the conclusions of your work are both valid and reliable? What steps do you think are the most challenging from a practical perspective? 
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
# > 
# > "Question 3" as described below only addresses **multiple linear regression**... but you'll also perhaps notice that "Question 3" as described above extends this to address addresses **logistic regression**.
#     
# ### Current Course Project Capability Level 
#     
# This homework's "Question 3" introduced the idea of performing some **multiple linear regression** analyses on dataset from the Canadian Social Connection Survey. While other questions of this homework focussed on other datasets, the general analyses and principles they introduce are no doubt informative and applicable to this the dataset for our course project. Ideally, this should put you in a position to quite proficiently perform **multiple linear regression** analyses for the course project if you so desire and find appropriate for the objectives of you course project submission. Thus, the following (and more) should be possible at this stage... 
# 
# 1. Select multiple **predictors predictor** from the Canadian Social Connection Survey data and examine how they jointly influence an outcome variable, paying special attention to the inclusion and interpretation of **categorical** and **indicator variables** and **interactions** (in terms of "baseline" reference groups and "contrast" or "offsets").
# 
# 2. Visualize different kinds of **predictive association** relationships, including **interactions** and relationship between **predictor** and the **outcome** variables that change across different levels of other **categorical** or **indicator predictor variables**, using tools like `plotly.express`.
# 
# 3. Use **coefficient hypothesis testing** and "in sample" versus "out of sample" **model performance** evaluation to perform **model building** and examine **generalizability** of **fitted models**.
#        
# 4. Assess the presence of **multicollinearity** by considering the **condition numbers** of **fitted models** (with "centering and scaling") and their subsequent potential implications on **generalizability** of **fitted models**; and, perhaps even examine **pairwise correlation** and/or **variance inflation factors** for each **predictor variable** if you're feeling extra ambitious and want to go well "above and beyond" (in which case you could also consider the relationship between **multicollinearity** and why one level of a **categorical** variable is always omitted).
# 
# 5. Compare and contrast such analyses and their benefits with previous methodologies introduced and considered in the course.
#     
# 6. Explore using model diagnostic to check assess the assumptions of your **multiple linear regression** analyses, and reflect on how failurse of these assumptions might impact the reliability of your findings and conlusions derived from your **fitted model**.
# 
# </details>
# 

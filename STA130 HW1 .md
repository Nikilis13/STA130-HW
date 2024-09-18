```python
#1
import pandas as pd
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-05/villagers.csv"
df = pd.read_csv(url) #<-- reads the url and stores it in the variable 'df' 

df.isna().sum() #<-- shows the missing data in every colomn
```




    row_n           0
    id              1
    name            0
    gender          0
    species         0
    birthday        0
    personality     0
    song           11
    phrase          0
    full_id         0
    url             0
    dtype: int64




```python
#2
import pandas as pd
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-05/villagers.csv"
df = pd.read_csv(url) #<-- reads the url and stores it in the variable 'df'

print('number of columns in the data set:' , df.shape[1]) #<-- prints the number of columns in the data set
print('number of rows in the data set:' , df.shape[0]) #<-- prints the number of rows in the data set
```

    number of columns in the data set: 11
    number of rows in the data set: 391



```python
#3
import pandas as pd
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-05/villagers.csv"
df = pd.read_csv(url) #<-- reads the url and stores it in the variable 'df'

print(df.describe()) #<-- summarizes the columns in the data set
print(df.value_counts()) #<-- shows the data description throughout the data set
```

                row_n
    count  391.000000
    mean   239.902813
    std    140.702672
    min      2.000000
    25%    117.500000
    50%    240.000000
    75%    363.500000
    max    483.000000
    row_n  id        name      gender  species   birthday  personality  song             phrase      full_id            url                                                               
    2      admiral   Admiral   male    bird      1-27      cranky       Steep Hill       aye aye     villager-admiral   https://villagerdb.com/images/villagers/thumb/admiral.98206ee.png     1
    317    olive     Olive     female  cub       7-12      normal       Cafe K.K.        sweet pea   villager-olive     https://villagerdb.com/images/villagers/thumb/olive.d5fcb11.png       1
    330    patty     Patty     female  cow       5-10      peppy        K.K. Calypso     how now     villager-patty     https://villagerdb.com/images/villagers/thumb/patty.3e17f7f.png       1
    328    pate      Pate      female  duck      2-23      peppy        K.K. Steppe      quackle     villager-pate      https://villagerdb.com/images/villagers/thumb/pate.c60838c.png        1
    327    pashmina  Pashmina  female  goat      12-26     uchi         K.K. Tango       kidders     villager-pashmina  https://villagerdb.com/images/villagers/thumb/pashmina.8916368.png    1
                                                                                                                                                                                             ..
    150    eloise    Eloise    female  elephant  12-8      snooty       K.K. Jazz        tooooot     villager-eloise    https://villagerdb.com/images/villagers/thumb/eloise.112208b.png      1
    149    elmer     Elmer     male    horse     10-5      lazy         K.K. Waltz       tenderfoot  villager-elmer     https://villagerdb.com/images/villagers/thumb/elmer.cc7df52.png       1
    148    ellie     Ellie     female  elephant  5-12      normal       Cafe K.K.        wee one     villager-ellie     https://villagerdb.com/images/villagers/thumb/ellie.5a144a6.png       1
    147    elise     Elise     female  monkey    3-21      snooty       Neapolitan       puh-lease   villager-elise     https://villagerdb.com/images/villagers/thumb/elise.aa507f1.png       1
    483    zucker    Zucker    male    octopus   3-8       lazy         Spring Blossoms  bloop       villager-zucker    https://villagerdb.com/images/villagers/thumb/zucker.8dbb719.png      1
    Name: count, Length: 379, dtype: int64



```python
#4
import pandas as pd
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url) #<-- reads the url and stores it in the variable 'df'

print(df.shape) #<-- shows us the total number of rows and columns, respectively
print(df.describe()) #<-- gives us the summary for all numeric columns ('count' shows us the number of non-missing values for each missing column)
print(df.describe(include='all')) #<-- gives us the summary for all columns irrespective of numerical value
print(df.select_dtypes(include=['number']).isna().sum()) #<-- shows us the number of missing values in each column
```

    (891, 15)
             survived      pclass         age       sibsp       parch        fare
    count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
    mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
    std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
    min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
    25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
    50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
    75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
    max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
              survived      pclass   sex         age       sibsp       parch  \
    count   891.000000  891.000000   891  714.000000  891.000000  891.000000   
    unique         NaN         NaN     2         NaN         NaN         NaN   
    top            NaN         NaN  male         NaN         NaN         NaN   
    freq           NaN         NaN   577         NaN         NaN         NaN   
    mean      0.383838    2.308642   NaN   29.699118    0.523008    0.381594   
    std       0.486592    0.836071   NaN   14.526497    1.102743    0.806057   
    min       0.000000    1.000000   NaN    0.420000    0.000000    0.000000   
    25%       0.000000    2.000000   NaN   20.125000    0.000000    0.000000   
    50%       0.000000    3.000000   NaN   28.000000    0.000000    0.000000   
    75%       1.000000    3.000000   NaN   38.000000    1.000000    0.000000   
    max       1.000000    3.000000   NaN   80.000000    8.000000    6.000000   
    
                  fare embarked  class  who adult_male deck  embark_town alive  \
    count   891.000000      889    891  891        891  203          889   891   
    unique         NaN        3      3    3          2    7            3     2   
    top            NaN        S  Third  man       True    C  Southampton    no   
    freq           NaN      644    491  537        537   59          644   549   
    mean     32.204208      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    std      49.693429      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    min       0.000000      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    25%       7.910400      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    50%      14.454200      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    75%      31.000000      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    max     512.329200      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    
           alone  
    count    891  
    unique     2  
    top     True  
    freq     537  
    mean     NaN  
    std      NaN  
    min      NaN  
    25%      NaN  
    50%      NaN  
    75%      NaN  
    max      NaN  
    survived      0
    pclass        0
    age         177
    sibsp         0
    parch         0
    fare          0
    dtype: int64



```python
#5
# an attribute is a property/characteristic which does not requre parentheses as it does not perform any sort of computational action and instead holds data directly. A notable example of this would be the 'df.shape' command, and we can see how it doesn't have any parentheses since all it's responsible for is the data it holds.
# a method is a function which belongs to an object and perfoms computational action when called, basically the opposite of an attribute. An example of a method would be 'df.describe()', the method needs to have parentheses because it needs to perform a series of calculations and hold those values until the function is called.
# in summary, attributes hold information or a value pertaining to the object whereas a method performs operations or computations on the object itself.
```


```python
#chatgpt link
# https://chatgpt.com/share/66e38646-bd2c-8008-afc6-efc73562d90c
```


```python
#6
# count: number of non-number (NaN) observations in the data set for each variable telling us how many entries exist for each column excluding any missing values.
# mean: the average value of the observations, calculated by dividing the sum of all the values by the number of values.
# std (standard deviation): measures the average distance of each data point from the mean.
# min: the smallest value in the dataset for each variable.
# 25% (First Quartile): The value below which 25% of obeservations fall i.e the lower end of the data set.
# 50% (Median): the middle value of the data set.
# 75% (Third Quartile): the value below which 75% of observations fall i.e the higher end of the data set.
# max: the largest value in the dataset for each variable.
```


```python
#7
# using df.dropna() would be prefered over del df['col'] when we need to remove rows/columns with missing values whereas the latter primarily focuses on removing specific columns.
# that being said, del df['col'] can be preferred over df.dropna() when we are looking to remove columns that have too many missing values i.e. targetting specific columns rather than removing all columns with missing values. Some instances, many would want to keep some columns with missing values for any reason they choose, that's where del df['col'] comes into play.
# using del df['col'] before df.dropna helps with data cleanliness because when we are clearling out unnecessary data, it helps to remove all the irrelevent columns first, so df.dropna() can more thoroughly clean out the remaining data.

import pandas as pd
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url) #<-- reads the url and stores it in the variable 'df'

# this is to show the missing data in the data set prior to being cleansed
print(df.head())
print(df.isna().sum())

columns_to_remove = ['id', 'song']

# using a for loop we can begin removing columns kinda like we are
for col in columns_to_remove:
    if col in df.columns:
        del df[col]

# this verifies the columns after removing all the unnessesary data
print(df.head())
print(df.isna().sum())

# prints the shape of the data set before and after cleaning
print('data set before cleaning ', df.shape)
print('data set after cleaning ', df.dropna().shape)

# shows the cleaned data frame
print("cleaned data frame: ", df.dropna().head())

# this method of cleaning the data would be the most efficiant way. Incorporating a for-loop insures the code is in the most efficiant standard, reducing the lines of code needed to perform the same operation. Showing the data set's before and after is also very key in seeing the data set cleaning itself.
```

       row_n       id     name  gender    species birthday personality  \
    0      2  admiral  Admiral    male       bird     1-27      cranky   
    1      3  agent-s  Agent S  female   squirrel      7-2       peppy   
    2      4    agnes    Agnes  female        pig     4-21        uchi   
    3      6       al       Al    male    gorilla    10-18        lazy   
    4      7  alfonso  Alfonso    male  alligator      6-9        lazy   
    
              song    phrase           full_id  \
    0   Steep Hill   aye aye  villager-admiral   
    1      DJ K.K.  sidekick  villager-agent-s   
    2   K.K. House   snuffle    villager-agnes   
    3   Steep Hill   Ayyeeee       villager-al   
    4  Forest Life  it'sa me  villager-alfonso   
    
                                                     url  
    0  https://villagerdb.com/images/villagers/thumb/...  
    1  https://villagerdb.com/images/villagers/thumb/...  
    2  https://villagerdb.com/images/villagers/thumb/...  
    3  https://villagerdb.com/images/villagers/thumb/...  
    4  https://villagerdb.com/images/villagers/thumb/...  
    row_n           0
    id              1
    name            0
    gender          0
    species         0
    birthday        0
    personality     0
    song           11
    phrase          0
    full_id         0
    url             0
    dtype: int64
       row_n     name  gender    species birthday personality    phrase  \
    0      2  Admiral    male       bird     1-27      cranky   aye aye   
    1      3  Agent S  female   squirrel      7-2       peppy  sidekick   
    2      4    Agnes  female        pig     4-21        uchi   snuffle   
    3      6       Al    male    gorilla    10-18        lazy   Ayyeeee   
    4      7  Alfonso    male  alligator      6-9        lazy  it'sa me   
    
                full_id                                                url  
    0  villager-admiral  https://villagerdb.com/images/villagers/thumb/...  
    1  villager-agent-s  https://villagerdb.com/images/villagers/thumb/...  
    2    villager-agnes  https://villagerdb.com/images/villagers/thumb/...  
    3       villager-al  https://villagerdb.com/images/villagers/thumb/...  
    4  villager-alfonso  https://villagerdb.com/images/villagers/thumb/...  
    row_n          0
    name           0
    gender         0
    species        0
    birthday       0
    personality    0
    phrase         0
    full_id        0
    url            0
    dtype: int64
    data set before cleaning  (391, 9)
    data set after cleaning  (391, 9)
    cleaned data frame:     row_n     name  gender    species birthday personality    phrase  \
    0      2  Admiral    male       bird     1-27      cranky   aye aye   
    1      3  Agent S  female   squirrel      7-2       peppy  sidekick   
    2      4    Agnes  female        pig     4-21        uchi   snuffle   
    3      6       Al    male    gorilla    10-18        lazy   Ayyeeee   
    4      7  Alfonso    male  alligator      6-9        lazy  it'sa me   
    
                full_id                                                url  
    0  villager-admiral  https://villagerdb.com/images/villagers/thumb/...  
    1  villager-agent-s  https://villagerdb.com/images/villagers/thumb/...  
    2    villager-agnes  https://villagerdb.com/images/villagers/thumb/...  
    3       villager-al  https://villagerdb.com/images/villagers/thumb/...  
    4  villager-alfonso  https://villagerdb.com/images/villagers/thumb/...  



```python
#8a
import pandas as pd
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url) #<-- reads the url and stores it in the variable 'df'

# displays the first few rows of the data set
print(df.head())

# check for missing values and colun names
print('missing values in each column', df.isna().sum())
print('column names and data types', df.dtypes)

# grouping by 'class' and descitibing 'age'
grouped_describe = df.groupby('class')['age'].describe()

print('descriptive statistics for "age" grouped by "class": ', grouped_describe)
```

       survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
    0         0       3    male  22.0      1      0   7.2500        S  Third   
    1         1       1  female  38.0      1      0  71.2833        C  First   
    2         1       3  female  26.0      0      0   7.9250        S  Third   
    3         1       1  female  35.0      1      0  53.1000        S  First   
    4         0       3    male  35.0      0      0   8.0500        S  Third   
    
         who  adult_male deck  embark_town alive  alone  
    0    man        True  NaN  Southampton    no  False  
    1  woman       False    C    Cherbourg   yes  False  
    2  woman       False  NaN  Southampton   yes   True  
    3  woman       False    C  Southampton   yes  False  
    4    man        True  NaN  Southampton    no   True  
    missing values in each column survived         0
    pclass           0
    sex              0
    age            177
    sibsp            0
    parch            0
    fare             0
    embarked         2
    class            0
    who              0
    adult_male       0
    deck           688
    embark_town      2
    alive            0
    alone            0
    dtype: int64
    column names and data types survived         int64
    pclass           int64
    sex             object
    age            float64
    sibsp            int64
    parch            int64
    fare           float64
    embarked        object
    class           object
    who             object
    adult_male        bool
    deck            object
    embark_town     object
    alive           object
    alone             bool
    dtype: object
    descriptive statistics for "age" grouped by "class":          count       mean        std   min   25%   50%   75%   max
    class                                                            
    First   186.0  38.233441  14.802856  0.92  27.0  37.0  49.0  80.0
    Second  173.0  29.877630  14.001077  0.67  23.0  29.0  36.0  70.0
    Third   355.0  25.140620  12.495398  0.42  18.0  24.0  32.0  74.0



```python
#8b
# df.describe() gives a broad overview of missing values and statistics across the entrire dataset, while df.groupby("col1")["col2"].describe() provides detailed statistics within each group, reflecting how missing values and other statistics vary by group.
```


```python
#8c
# overall I found quicker explanations with ChatGPT with more thorough examples than Google Search. Both are really good at debugging different errors within the code but when it comes down to accuracy and speed, I feel like ChatGPT had the upper hand, especially because the explanations the bot replied with such as the diagrams or snippets of the code itself really helped me with troubleshooting and finding errors in my code, as well as understanding exactly what errors i'm making and why.

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
#9
# yes
```


```python
# chatgpt link
# https://chatgpt.com/share/66e3a9f0-6dcc-8008-8757-d8fa64d58ee4
```


```python

```

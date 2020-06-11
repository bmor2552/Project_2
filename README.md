# DS Borg Homes 

![Shacking hands in front of homes](https://media-public.canva.com/MADQ5ew6_QY/1/thumbnail_large-1.jpg)

**The Goal:**
 The goal of this project is to predict the sale price of homes based off of the zip code the home is located in.

**The Problem:**

During our daily commute we tend to drive past homes that are either appealing to the eye or homes that just need a bit of work. After that moment of admiration or of making plans on how you would fix up a home, you drive on and forget about it.

But what if you didn't forget about it, what if you actually followed through with your idea and bought a brand new home, or even a fixer upper? Where would you even begin to bring such a thought to life? Most importantly, how much would it be to make it happen?

Once those questions are answered you will see that the more research you do the more the prices vary depending on the area you decide to purchase in. Not to mention the other factors that play a role in a homes price i.e. water front view and a basement. So now the price you had set for a budget begins to increase due to the lack of knowledge when looking into the value of the home you want to purchase.

**The Solution:**

Here at DS Borg Homes our team has spent the last year developing a model that will be able to tell how much a home is actually worth by just simply typing in the zip code! Voilà!! No more long nights of scouring through papers, websites, or phone numbers trying to find the answers you need. With our model, DS Borg Homes will help you find the perfect price for the perfect home.

**The Data:** 
  The data used to achieve our goal is from the King County House Sales dataset.


*The following link is the csv file of the original dataset we used:*

https://github.com/bmor2552/Project_2/blob/master/Data/kc_house_data.csv


**The Plan:**
  The way we plan on tackling this is by doing the following:

1. Cleaning the data
2. Exploring the data
3. Modeling the data


# Cleaning & Exploring The Data:

The first thing we did was look at the data while keeping in mind the following questions:

1. What information do we have?
2. What information are we missing?
3. What information do we need to get rid of?

From viewing the data we found that we had 22 columns which consisted of information on a list of homes that played a role in its price for the years 2014 and 2015; size of the home, the year it was built, the year it was renovated, number of bathrooms and bedrooms, etc.

We also found that the missing values were represented with a ? or a NaN, this was the issue we decided to deal with first. After the missing values were taken care of we continued viewing the data for anything that would interfere with the modeling process. This included changing outliers and dropping columns. Afterwards the clean data was saved to a csv file, to work with.


*The following csv file is where the cleaned data of the needed columns are found:*

https://github.com/bmor2552/Project_2/blob/master/Data/clean_data.csv


*The following notebook shows the cleaning & exploring of the data as well as our findings from doing so:*

 https://github.com/bmor2552/Project_2/blob/master/Jupyter%20Notebooks/STEP_1_Cleaning%20and%20Exploring%20Data.ipynb


# Modeling The Data

When working with predictive Linear Regression Models you must first choose how you will measure the accuracy of your work. For this instance we chose to use the R-squared.

This R-squared score is used in statistics as a measurement of how close you can get your data to fit on a regression line. In other words, the closer you can get your data to fit on that line the better your model is doing at predicting the sale price of a home. The highest score you can get is a 1, in percentages that is 100% of accuracy. We chose this scoring metric because we knew that several models would be tested and the R-squared score would be compatible with those test models. The next step was to run our first model, our Baseline Model.


*You can find more information on the R-squared score using the link below:*

https://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit


# Baseline Model vs Final Model

**Baseline**

The baseline model we chose to start with was a Linear Regression Model, this is used in statistics to compare the dependent variable to the independent variables. This model was just to see how the cleaned data would perform before making any changes. Our *price* column was the dependent variable that was affected by and change in the other columns which were the dependent variables (assuming there are no columns correlated with one another aka interacting).

You to make sure there is a leveled playing field for your model, so to make sure the range of numerical values in our data would affect our model we chose a scaler, a Robust Scaler. This would help numerical spread of the numbers which helps with outliers. If you know anything about outliers, then you know how the model will take them in and give results that are off. From there we ran the model and got a score of .70, remember the goal is to get our R-squared score as close to 1 as possible. 


*The following link will provide more information on Linear Regression:*

https://en.wikipedia.org/wiki/Linear_regression


**Improving The Model**

Even though our scaler was brought in to avoid any issues with the spread of the data the model was still being affected. So a heat map was plotted to see which columns were interacting with one another. From the heat map we saw that although the columns were independent from the *price* they were dependent on one another, in other words correlated.

To take care of this we cleaned up the outliers for the most correlated columns. We then binned and encoded the columns we assumed were important (this assumption was based off of the heat map analysis). This methods we tried were known as feature engineering; target encoder and one-hot encoder.


*The following link provides more information on Feature Engineering:*

https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114


**Transforming The Data**
The focus for transforming the data was to get a normal distributing of the data, this way the regression model can perform better. The data transformer we chose was Quantile Transform, which helps us reach the goal of normal distribution by reducing the effects of our outliers in the data by focusing on quantiles. Why did we use this, simply put it will take care the columns that are interacting with one another. This effort was short lived because once we ran our baseline model on the transformed data, we got an R-squared score of .60. So it was time to try a new model. 


*The following link provides the documentation for Quantile Transform:*

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html


**The Search of a New Model** 
Maybe we made things more complicated then it need to be by adding the features that we did, so we gave two models a try, Ridge and Lasso regression. Both Lasso and Ridge are simple regressions used as simplified models which assist in the prevention of over-fitting your data aka doing way too much engineering. 


*Lasso*

When working with this model we adjusted the alpha as followed: 0.001, 0.1, 1000. All three alphas gave us an R-squared score of .70. We were back at the beginning. This model uses your data to drop the unimportant columns and then runs its prediction. 


*You can find more information on Lasso using the following link:* 

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?highlight=lasso#sklearn.linear_model.Lasso


*Ridge*

With this model we adjusted the alpha as followed: 0.001, 0.1, 1000. The first two alphas gave us an R-squared score of .70, and the last one gave us a score of .68. Our progress was in limbo! With this model at every alpha level we tried, the coefficient of the data was changed in order to improve the model, but this caused us to do worse because we lost too much data. In simple terms the multicollinearity of our columns cause use to lose more data we indented to get rid of.


*You can find more information on Ridge using the following link:*

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge


Comparing the models to baseline: Below is a table that reflects the results of each model we ran up until this point.


![](https://raw.githubusercontent.com/ezgigm/Project_2/Ezgi/Screen%20Shot%202020-03-27%20at%2010.58.36%20AM.png)


**Final Model**


*GAM*

GAM is also known as Generalized Additive Model, this model is used in statistics for its ability to interpret and regulate your data. In simple terms it is flexible enough to handle the multicollinearity our data has. To improve the performance of GAM we used tensored both the latitude and longitude columns, added gamma distribution in combination with log link. The R-squared score increased drastically from .70 to .90.


*The following link will provide more information on GAM:* 

https://pygam.readthedocs.io/en/latest/


*The following notebook shows the modeling process of the data as well as our findings from doing so:*

https://github.com/bmor2552/Project_2/blob/master/Jupyter%20Notebooks/STEP_2_Modeling.ipynb


# Findings

The R-squared score we received using the GAM Model vs our Baseline Model: Linear Regression Model was better due to the fact that GAM was able to handle our independent variables correlating with one another.   

Our goal was to find a model that would be able to predict the price of a home based off of the area the home was located in. We took modeling a step further to accomplish this by selecting zip codes for our GAM Model to test. 


The chart below shows the results of our test per zip code:

![](https://raw.githubusercontent.com/ezgigm/Project_2/Ezgi/Screen%20Shot%202020-03-27%20at%209.18.26%20AM.png)


As you can see the model did an amazing job predicting the price of a home! Based on the areas listed in the chart above, if you were relocating and needed to buy a home but wanted more bang for your buck you could use this model to find a price that fits in your budget. 

If you are already in that area and looking to buy a fixer upper you could use this model to find a home that is cheap enough to be fixed up and profit from. 

The possibilities are endless!!!


# Future Improvements

In the future our team plans to perfect the baseline model by reproaching feature engineering using different methods from what we used in our first round of model testing. 

We as well would like to build a model for every zip code in addition to subsets of zip codes i.e. dividing larger areas into smaller areas. To further analyze pricing of homes we will look into things such as having an ocean view vs having a man-made lake view.  

When it comes to the data in the future, we will collect more recent data past the year 2015 that includes data after the COVID-19 spread and recovery phase is over. 


# Repository Guide

## CSV Files
 
 Original csv file of dataset

https://github.com/bmor2552/Project_2/blob/master/Data/kc_house_data.csv

Column names & their description

https://github.com/bmor2552/Project_2/blob/master/Data/ColumnNames.md


Cleaned csv file of data set

https://github.com/bmor2552/Project_2/blob/master/Data/clean_data.csv


## Notebooks
 
 **Cleaning & exploring the data and creating csv file** 
 
 https://github.com/bmor2552/Project_2/blob/master/Jupyter%20Notebooks/STEP_1_Cleaning%20and%20Exploring%20Data.ipynb
 
 
 **Modeling the data**
 
 https://github.com/bmor2552/Project_2/blob/master/Jupyter%20Notebooks/STEP_2_Modeling.ipynb
 
 
 ## Photos for READ.me
 
 https://github.com/bmor2552/Project_2/blob/master/README_ScreenShots/Screen%20Shot%202020-03-26%20at%208.08.58%20PM.png
 

## Presentation

https://github.com/bmor2552/Project_2/blob/master/DS%20Borge%20Homes.pdf

 

# Additional Resources

Information on original dataset

https://info.kingcounty.gov/assessor/DataDownload/default.aspx


# Team Members

**Takehiro Yasuoka:** https://github.com/Tyasuoka

**Ezgi Gumusbas:** https://github.com/ezgigm

**Boi Moriba:** https://github.com/bmor2552?tab=stars





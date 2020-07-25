# IAcapstone
This is the final report for capstone project of IBM Advanced Data Science Course.

# Use Case
I work in a goberment body in a region of Spain that provides services and IT assistance to farmers. We are specially focused on developing decission support tools for different agricultural purposes. In one of this tools, it would be in a great help to have a model that could predict the next year land usage for a given area.

The idea of the project is that the farmer uses his land in a cyclical way, using patterns, if we can recognize these patterns, we could know which crop is the most likely to use the following year, or the 2-3 crops with the most probability.
This cyclical use of crops is a common practice to leave the land uncultivated one year out of three, to avoid problems with pests, fertilizers, etc. In other cases, there are static crops that generally do not vary (trees, olive, vineyard).
The model does not intend to train a specific case of a farmer, we try to see if, taking a very high number of use cases, we can extract the generality of the patterns of use of the crops and given a sequence of years, we can know what the most likely crop for the following year.

As a first attempt, we gathered the last 9 years history for around 2M points tagging for each year the crop code, hoping that we can apply some LSTM network to predict the next year crop code. 

# Data Set
The data consist in around 2 million points scatterred over all the region territory, the original data file was a shape file with point features, having each point an attribute for each year with the crop code used.
We have 27 different crops: WHEAT,CORN,BARLEY,FLOOR,SUNFLOWER,RAPE,GREEN PEAS,ALFALFA,FORAGE,BEET,VINEYARD,OLIVE,HORTICULTURAL,AROMATIC,FRUITS,SCRUB,DIFFERENT KINDS OF LEAFY TREES, etc
All this data comes from claims for payment of CAP subsidies, accesible for us as regional goverment agency, but it cannot be shared, but an small sample without geolocalization is provided as example.

# ETL and Feature Creation

Data Exploration and Data Visualization 
# Data Quality Assessment

Model Definition and Training
Model Evaluation, Tuning, Deployment and Documentation




Feature Engineering

Selection and justification of Model Performance Indicator

At least one traditional Machine Learning Algorithm and one DeepLearning Algorithm applied and demonstrated

Model performance between different feature engineerings and models compared and documented
Please assign one point for each item below which is properly covered in the ADD

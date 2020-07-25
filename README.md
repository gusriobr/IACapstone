# IAcapstone
This is the final report for capstone project of IBM Advanced Data Science Course.

# Use Case
I work in a goberment body in a region of Spain that provides services and IT assistance to farmers. We are specially focused on developing decission support tools for different agricultural purposes. In one of this tools, it would be in a great help to have a model that could predict the next year land usage for a given area.

The idea of the project is that the farmer uses his land in a cyclical way, using patterns, if we can recognize these patterns, we could know which crop is the most likely to use the following year, or the 2-3 crops with the most probability.
This cyclical use of crops is a common practice to leave the land uncultivated one year out of three, to avoid problems with pests, fertilizers, etc. In other cases, there are static crops that generally do not vary (trees, olive, vineyard).
The model does not intend to train a specific case of a farmer, we try to see if, taking a very high number of use cases, we can extract the generality of the patterns of use of the crops and given a sequence of years, we can know what the most likely crop for the following year.

As a first attempt, we gathered the last 9 years history for around 2M points tagging for each year the crop code, hoping that we can apply some LSTM network to predict the next year crop code. 

# Data Set, ETL and Feature Creation
The data consist in around 2 million points scatterred over all the region territory, the original data file was a shape file with point features, having each point an attribute for each year with the crop code used.
We have 27 different crops: WHEAT,CORN,BARLEY,FLOOR,SUNFLOWER,RAPE,GREEN PEAS,ALFALFA,FORAGE,BEET,VINEYARD,OLIVE,HORTICULTURAL,AROMATIC,FRUITS,SCRUB,DIFFERENT KINDS OF LEAFY TREES, etc

All this data comes from claims for payment of CAP subsidies, accesible for us as regional goverment agency, but it cannot be shared in the project, but I thin the data exploration and visualization notebook gives a sufficient idea of the data structure.

# Data Exploration, Visualization and Quality Assessment



# Model Definition and Training
Se ha intentando modeler directamente con un árbol de decisión para tomarlo como modelo base y ver si aumentan.

El conjunto de datos es muy elevado, el entrenamiento se ha hecho con un Ubuntu laptop con gpu (aunque debido a restricciones de keras en algunos casos los modelos lstm no hacen uso de gpu) 

LSTM and CONV modeling the the crop series


# Model Evaluation, Tuning, Deployment and Documentation
El modelo lstm es muy útil par aseries, pero en este caso, no es bastante, debido en mi opinión a la longitdu de la secuencia. Estos modelos se basan en aprender del contexto, pero en este caso el contexto es demasiado corto.
se ha intentado crar características adicionales, pero no ha mejorado


Para la optimización, en lugar de utilizado gridsarc . otra alternativa podría ser el uso de algoritmos genéticos


Feature Engineering

Selection and justification of Model Performance Indicator

At least one traditional Machine Learning Algorithm and one DeepLearning Algorithm applied and demonstrated

Model performance between different feature engineerings and models compared and documented
Please assign one point for each item below which is properly covered in the ADD

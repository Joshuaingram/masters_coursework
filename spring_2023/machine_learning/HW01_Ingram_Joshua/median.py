import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedShuffleSplit,
)
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    FunctionTransformer,
    OrdinalEncoder,
    MinMaxScaler,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import (
    make_column_transformer,
    make_column_selector,
    ColumnTransformer,
    TransformedTargetRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.stats import randint
from scipy import stats
import joblib

# Read CSV into pd dataframe
housing = pd.read_csv("housing.csv")

## Take a Quick Look at the Data Structure

# view first 5 rows of housing data
print(housing.head())

# info on data frame structure and data types
print(housing.info())

# counts by category for ocean_proximity variable
print(housing["ocean_proximity"].value_counts())

# summary of numerical attributes in dataframe
print(housing.describe())

# histograms with 50 bins of numerical attributes with 12 by 8 sizing
print("\n*** Writing histogram ***")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing.hist(bins=50, ax=ax)
fig.savefig("hist_numerical_attr.png")
plt.show()

## Create a Test Set


# input: data - dataframe, test_ratio - desired proportion of original data to be test data
# output: training dataframe and test dataframe
# This function randomly selects indices for the training and test data for the given data and ratio
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# define training and test datasets
train_set, test_set = shuffle_and_split_data(housing, 0.2)
print(len(train_set))

print(len(test_set))

# set seed for rng
np.random.seed(1)


# functions to maintain stability in selecting training and test data
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# add index column
housing_with_id = housing.reset_index()
# define new training and test sets with above defined functions using index column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# define Id using latitude and longitude of districts, define training and test sets
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

# SK-Learn function similar to shuffle_and_split_data() function
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# create income category attribute with 5 cateogies
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# barplot showing frequencies of each income category
print("\n*** Writing barplot ***")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True, ax=ax)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
fig.savefig("barplot_inc_cat.png")
plt.show()

# generate 10 different splits of the same dataset
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
# for every index in the train and test sets
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# select first split
strat_train_set, strat_test_set = strat_splits[0]

# shorter way to get single split
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

# proportions of income categories in test set
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# drop the income_cat column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


## Explore and Visualize the Data to Gain Insights

# copy original training set
housing = strat_train_set.copy()

### Visualizing Geographical Data

# create and save scatterplot of districts using lat and long
print("\n*** Writing scatterplot ***")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, ax=ax)
fig.savefig("scatter_lat_lon.png")
plt.show()

# change opaqueness of points to see density
print("\n*** Writing scatterplot ***")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2, ax=ax)
fig.savefig("scatter_lat_lon_dens.png")
plt.show()

# scatterplot of districts where colors indicate median house value and size indicate population size
print("\n*** Writing scatterplot ***")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    grid=True,
    s=housing["population"] / 100,
    label="population",
    c="median_house_value",
    cmap="jet",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7),
    ax=ax,
)
fig.savefig("scatter_lat_lon_med_pop.png")
plt.show()


## Look for Correlations

# corrleation matrix (pearson's) between all atrributes
corr_matrix = housing.corr()

print(corr_matrix["median_house_value"].sort_values(ascending=False))

# plot scatterplot matrix of all selected attributes
attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]
print("\n*** Writing scatter matrix ***")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
scatter_matrix(housing[attributes], figsize=(12, 8), ax=ax)
fig.savefig("scatter_matrix_plot.png")
plt.show()

# plot scatterplot between median_income and median_house_value
print("\n*** Writing scatterplot ***")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing.plot(
    kind="scatter",
    x="median_income",
    y="median_house_value",
    alpha=0.1,
    grid=True,
    ax=ax,
)
fig.savefig("scatter_plot_inc_house.png")
plt.show()

## Expirment With Attribute Combinations

# define new attributes
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# correlation matrix with new attributes
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


## Prepare the Data For Machine Learning Algorithms

# separate predictors from target
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


### Clean the Data

# removing missing features
# option 1 - get rid of corresponding districts using dropna
# housing.dropna(subset=["total_bedrooms"], inplace=True)    # option 1

# option 2 - drop entire attribute using drop
# housing.drop("total_bedrooms", axis=1)       # option 2

# option 3 - set missing values to median using fillna
# median = housing["total_bedrooms"].median()  # option 3
# housing["total_bedrooms"].fillna(median, inplace=True)

# imputer in SK learn... imputes using median
imputer = SimpleImputer(strategy="median")

# select only numerical attributes
housing_num = housing.select_dtypes(include=[np.number])

# fit imputer on training data to get medians
imputer.fit(housing_num)

# median of each attribute
print(imputer.statistics_)

print(housing_num.median().values)

# transform training set with imputer fit
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


## Handling Text and Categorical Attributes

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))

# convert text categories to ordinal
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print(housing_cat_encoded[:8])

# list of categories
print(ordinal_encoder.categories_)

#  one hot encode categorical variables (dummy attributes)
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

print(housing_cat_1hot)

print(housing_cat_1hot.toarray())

print(cat_encoder.categories_)

# convert categorical feature into one-hot representation
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)

print(cat_encoder.transform(df_test))

df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
print(pd.get_dummies(df_test_unknown))

# handle unknown category with zeros
cat_encoder.handle_unknown = "ignore"
print(cat_encoder.transform(df_test_unknown))

# columns names in featuer name attribute, ensures any dataframe fed to this estimator has same column names
print(cat_encoder.feature_names_in_)

print(cat_encoder.get_feature_names_out())


## Feature Scaling amd Transformation

# normalize attributes (0-1)
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# standardize values
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# create new attribute using a radial basis function for housing_median_age
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# scale the lavbels using standardization
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# train simple linear regression model on scaled labels (median_income)
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
# new data
some_new_data = housing[["median_income"]].iloc[:5]

# make predictions with fit model
scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# automatically scale lavels and train the regression model on scaled labels
model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)


## Custom Transformers

# log transformation for population since it is skewed
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

# Gassian RBF transform housing_median_age
rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.0]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

# add feature that measures the geopgrahic similarity between each district and san francisco
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

# compute ratio between the input features 0 and 1
ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
print(ratio_transformer.transform(np.array([[1.0, 2.0], [3.0, 4.0]])))


# custom transformer that acts similar to StandardScaler'
class StandardScalerClone(BaseEstimator, TransformerMixin):
    # no *args or *kwargs
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    # y is required even though we don't use it
    def fit(self, X, y=None):
        # checks that X is an array with finite float values
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # every estimator stores this in fit()
        self.n_features_in_ = X.shape[1]
        # always return self
        return self

    def transform(self, X):
        # looks for learned attributes (with trailing _)
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


# transformer that uses Kmeans clusterer to identify main clusters, measure how similar each sample is to each cluster center
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# find 10 clusters, using lat and long weighing each by their median house value
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
similarities = cluster_simil.fit_transform(
    housing[["latitude", "longitude"]], sample_weight=housing_labels
)

print(similarities[:3].round(2))

################
# Used code from online textbook python notebooks to recreate the plot
################
housing_renamed = housing.rename(
    columns={
        "latitude": "Latitude",
        "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)",
    }
)
housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing_renamed.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    grid=True,
    s=housing_renamed["Population"] / 100,
    label="Population",
    c="Max cluster similarity",
    cmap="jet",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7),
    ax=ax,
)
plt.plot(
    cluster_simil.kmeans_.cluster_centers_[:, 1],
    cluster_simil.kmeans_.cluster_centers_[:, 0],
    linestyle="",
    color="black",
    marker="X",
    markersize=20,
    label="Cluster centers",
)
plt.legend(loc="upper right")
fig.savefig("scatter_density_clustering.png")
plt.show()


## Transformation Pipelines

# sklearn pipline class that helps with ordering of transformation sequence
num_pipeline = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ]
)

# name transformers using names of transformer classes
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# output of first two rows from pipeline transformation
housing_num_prepared = num_pipeline.fit_transform(housing_num)
print(housing_num_prepared[:2].round(2))

# get nice dataframe with get_feature_names_out() method
# df_housing_num_prepared = pd.DataFrame(
#    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
#    index=housing_num.index)

# numerical and categorical attribute names
num_attribs = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
cat_attribs = ["ocean_proximity"]

# transformation pipeline for categorical attributes
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
)

# combine numerical and categorical pipelines
preprocessing = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ]
)

# select all features of given types (dtype_include) to make the proper columns transformations
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

# apply column transformations to housing dataset
housing_prepared = preprocessing.fit_transform(housing)


# build pipeline that does all transformations in one go
# create new attributes with ratios (e.g. rooms per house in a give district)
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


# impute with median, log transform, cluster, make into numerical pipeline
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler(),
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# column transformation, applying each pipeline (listed) to designated columns
preprocessing = ColumnTransformer(
    [
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        (
            "log",
            log_pipeline,
            [
                "total_bedrooms",
                "total_rooms",
                "population",
                "households",
                "median_income",
            ],
        ),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline,
)

# apply transformation pipeline to housing data
housing_prepared = preprocessing.fit_transform(housing)

print(housing_prepared.shape)

print(preprocessing.get_feature_names_out())


## Select and Train a Model


### Training and Evaluating on the Training Set

# train linear regression model
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

# predictions
housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_labels.iloc[:5].values)

# rmse on training set
lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(lin_rmse)

# train decision tree
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

# predictions for training set
housing_predictions = tree_reg.predict(housing)


### Better Evaluation Using Cross-Validation

# 10-fold cross-validation
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(tree_rmse)
# calculate rmse
tree_rmses = -cross_val_score(
    tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10
)

print(pd.Series(tree_rmses).describe())

# train random forest model
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
# calculate rmse
forest_rmses = -cross_val_score(
    forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10
)

print(pd.Series(forest_rmses).describe())


## Fine Tune Your Model


### Grid Search

# search for best combination of hyperparameter values for the random forest model
full_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ]
)
param_grid = [
    {
        "preprocessing__geo__n_clusters": [5, 8, 10],
        "random_forest__max_features": [4, 6, 8],
    },
    {
        "preprocessing__geo__n_clusters": [10, 15],
        "random_forest__max_features": [6, 8, 10],
    },
]
grid_search = GridSearchCV(
    full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error"
)
grid_search.fit(housing, housing_labels)

print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
print(cv_res.head())


### Randomized Search

# random search for hyperparameter tuning
param_distribs = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features": randint(low=2, high=20),
}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)

rnd_search.fit(housing, housing_labels)


### Analyzing the Best Models and Their Errors

# calculate measure of relative importance of each attribute in making predictions
final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)

# sort measures of importance
print(
    sorted(
        zip(feature_importances, final_model["preprocessing"].get_feature_names_out()),
        reverse=True,
    )
)


### Evaluate Your System on the Test Set

# predictors and response for test set
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# predictions with final_model for test set
final_predictions = final_model.predict(X_test)

# test rmse
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)

# calculate confidence interval for rmse
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print(
    np.sqrt(
        stats.t.interval(
            confidence,
            len(squared_errors) - 1,
            loc=squared_errors.mean(),
            scale=stats.sem(squared_errors),
        )
    )
)


## Launch, Monitor, and Maintain Your System

# save model using joblib library
joblib.dump(final_model, "my_california_housing_model.pkl")

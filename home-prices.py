# Importing Libraries
import pandas as pd
import numpy as np
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.regression
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import norm, skew
from scipy import stats

#############################################################################################################################
# importing training and test datasets
train = pd.read_csv('train.csv', na_values = ('', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
    '1.#IND', '1.#QNAN', 'N/A', 'NULL', 'NaN', 'n/a', 'nan',
    'null'), keep_default_na = False)
test = pd.read_csv('test.csv', na_values = ('', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
    '1.#IND', '1.#QNAN', 'N/A', 'NULL', 'NaN', 'n/a', 'nan',
    'null'), keep_default_na = False)

SalePriceIndex = pd.read_csv('Ames_price_index.csv')



#############################################################################################################################
# Cleaning up data 
# Appending training and test data for cleanup
train_and_test = train.append(test, sort = False, ignore_index = True)

# Creating a rating vector based on description file and mapping values to numeric
data_dictionary = ({# Fields having "NA"
                  "MSZoning": {"NA": "None"},"Alley": {"NA": "None"}, "Exterior1st": {"NA": "None"}, "Exterior2nd": {"NA": "None"}, "MasVnrType": {"NA": "None"}, "GarageYrBlt": {"NA": "None"}, "MiscFeature": {"NA": "None"}, "SaleType": {"NA": "None"}, "Street": {"Grvl": 0, "Pave": 1}, "GarageType": {"NA": 0, "Detchd": 1, "CarPort": 2, "BuiltIn": 3, "Basment": 4, "Attchd": 5, "2Types": 6},      # Needs Hot encoding
   # Needs Hot encoding
                  # will fill them using function below "LotFrontage": {"NA": 0},
                  "MasVnrArea": {"NA": 0}, "BsmtFinSF1": {"NA": 0},"BsmtFinSF2": {"NA": 0},"BsmtUnfSF": {"NA": 0}, "TotalBsmtSF": {"NA": 0}, "BsmtFullBath": {"NA": 0},"BsmtHalfBath": {"NA": 0}, "GarageCars": {"NA": 0}, "GarageArea": {"NA": 0},
                  
                  "Utilities": {"NA": 0, "ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
                  "BsmtQual": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "BsmtCond": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "BsmtExposure": {"NA": 0, "No": 1, "Mn": 2, "Gd": 4, "Av": 3, "Ex": 5},
                  "BsmtFinType1": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
                  "BsmtFinType2": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
                  "Electrical": {"NA": 0, "Mix": 1, "FuseP": 2, "FuseF": 3, "FuseA": 4, "SBrkr": 5},
                  "KitchenQual": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "Functional": {"NA":0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
                  "FireplaceQu": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "GarageQual": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "GarageCond": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "PoolQC": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "Fence": {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
                  "ExterQual": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "ExterCond": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "HeatingQC": {"NA": 0, "Po": 1, "Fa": 2, "Gd": 4, "TA": 3, "Ex": 5},
                  "LotShape": {"IR3": 1, "IR2": 2, "IR1": 4, "Reg": 5},
                  "LandContour": {"Low": 1, "HLS": 2, "Bnk": 3, "Lvl": 4},
                  "LotConfig": {"FR3": 1, "FR2": 2, "CulDSac": 3, "Corner": 4, "Inside": 5},
                  "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
                  "GarageFinish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
                  # Additional fields below
                  "CentralAir": {"N": 0, "Y": 1},
                  "PavedDrive": {"N": 0, "P": 1, "Y": 2}})

# Use rating vestor to replace ratings
train_and_test.replace(data_dictionary, inplace = True)

# Ames, Iowa price index
SalePriceIndex['_YrMo'] = SalePriceIndex['YrSold'].astype(str) + SalePriceIndex['MoSold'].astype(str)
SalePriceIndex.drop(['YrSold'], axis = 1, inplace = True)
SalePriceIndex.drop(['MoSold'], axis = 1, inplace = True)
data_dictionary = dict(SalePriceIndex[['_YrMo', 'SalePriceIndex']].values)


train_and_test['_YrMo'] = train_and_test['YrSold'].astype(str) + train_and_test['MoSold'].astype(str)
# train_and_test.drop(['YrSold'], axis = 1, inplace = True)
# train_and_test.drop(['MoSold'], axis = 1, inplace = True)
train_and_test['_SalePriceIndex'] = train_and_test._YrMo.replace(data_dictionary)
train_and_test.drop(['_YrMo'], axis = 1, inplace = True)

# Lot Frontage
train_and_test['LotFrontage'].replace('NA',np.NaN, inplace = True)
train_and_test['LotFrontage'] = train_and_test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# MSZoning
train_and_test['MSZoning'].replace('None',np.NaN, inplace = True)
train_and_test['MSZoning'] = train_and_test['MSZoning'].transform(lambda x: x.fillna('RL'))

# SaleType
train_and_test['SaleType'].replace('None',np.NaN, inplace = True)
train_and_test['SaleType'] = train_and_test['SaleType'].transform(lambda x: x.fillna('WD'))

# GarageYrBlt
for x in range(train_and_test.shape[1]):
    if train_and_test['GarageYrBlt'][x] == 'None':
        if train_and_test['GarageArea'][x] > 0.0:
            train_and_test['GarageYrBlt'][x] = train_and_test['YearBuilt'][x] 


# Convert numeric variables to numeric datatype
train_and_test['LotFrontage'] = pd.to_numeric(train_and_test['LotFrontage'])
train_and_test['TotalBsmtSF'] = pd.to_numeric(train_and_test['TotalBsmtSF'])
train_and_test['GarageCars'] = pd.to_numeric(train_and_test['GarageCars'])
train_and_test['BsmtFullBath'] = pd.to_numeric(train_and_test['BsmtFullBath'])
train_and_test['GarageArea'] = pd.to_numeric(train_and_test['GarageArea'])
train_and_test['BsmtHalfBath'] = pd.to_numeric(train_and_test['BsmtHalfBath'])
train_and_test['BsmtFinSF1'] = pd.to_numeric(train_and_test['BsmtFinSF1'])
train_and_test['BsmtFinSF2'] = pd.to_numeric(train_and_test['BsmtFinSF2'])
train_and_test['MasVnrArea'] = pd.to_numeric(train_and_test['MasVnrArea'])
train_and_test['BsmtUnfSF'] = pd.to_numeric(train_and_test['BsmtUnfSF'])
# Convert string variables to string datatype
train_and_test['MSSubClass'] = train_and_test['MSSubClass'].apply(str)
train_and_test['YrSold'] = train_and_test['YrSold'].astype(str)
train_and_test['MoSold'] = train_and_test['MoSold'].astype(str)

# Remove out of bound rows
train_and_test.drop(train_and_test[(train_and_test['GrLivArea'] > 4000) & (train_and_test['SalePrice'] < 200000)].index, inplace = True)
train_and_test.drop(train_and_test[(train_and_test['OverallQual'] < 5) & (train_and_test['SalePrice'] > 200000)].index, inplace = True)
train_and_test.reset_index(drop=True, inplace=True)


train_and_test = pd.get_dummies(train_and_test,columns = ['MSSubClass', 'MSZoning', 'Alley', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
        'GarageYrBlt', 'MiscFeature', 'SaleType', 'SaleCondition', 'YrSold', 'MoSold' ])
    
    

#############################################################################################################################
# Split training and test data and copy for temp variables for analysis
analyse_test = train_and_test.iloc[1457:,:]
analyse_train = train_and_test.iloc[:1457,:]


##########################################
#@###   DEFLATED SALEPRICE BY SALEPRICEINDEX
analyse_train['_DeflatedSalePrice'] = analyse_train['SalePrice'] * analyse_train['_SalePriceIndex']






######
############################################

analyse_train_y = np.log(analyse_train['SalePrice'])

# drop emtpy SalePrice column in test set
analyse_test.drop(['SalePrice'], axis = 1, inplace = True)
analyse_train.drop(['SalePrice'], axis = 1, inplace = True)


############################
#### for deflated saleprice
analyse_train.drop(['_DeflatedSalePrice'], axis = 1, inplace = True)
analyse_train.drop(['_SalePriceIndex'], axis = 1, inplace = True)
analyse_test.drop(['_SalePriceIndex'], axis = 1, inplace = True)

# drop Id column in both test and train
analyse_test.drop(['Id'], axis = 1, inplace = True)
analyse_train.drop(['Id'], axis = 1, inplace = True)

##################
## Combine both for data transform
analyse_train = analyse_train.append(analyse_test, sort = False, ignore_index = True)
##################
#############################################################################################################################
# Evaluating collinearity
# Analysing train
# finding columns that contain 'NA' / these columns need to be addressed first before moving to other categorical variables
# =============================================================================
# a = pd.DataFrame
# for col in analyse_train.columns:
#     for item in enumerate(analyse_train[col].value_counts().index):
#         a = np.array(item[1], col)
#         
# analyse_train['new'] = analyse_train.apply(lambda x: ','.join(x[x.isnull()].index),axis=1)
# 
# a['new'] = (df.isnull() * df.columns.to_series()).apply(','.join,axis=1).str.strip(',')
#             
#             
# for col in train_and_test.columns:
#     for item in enumerate(train_and_test[col].value_counts().index):
#         if item[1] == "None":
#             print (item[1], col)
# =============================================================================
# All NA values have been replaced
  
# remove non-numeric columns          
## analyse_train = analyse_train._get_numeric_data()
# all non-numeric columns removed; removed 21 columns


# Adding Intercept **
# analyse_train = add_constant(analyse_train)


#  try out removing some variables
######       analyse_train.drop(['CentralAir'], axis = 1, inplace = True)
# analyse_train.drop(['GarageType'], axis = 1, inplace = True)
######      analyse_train.drop(['EnclosedPorch'], axis = 1, inplace = True)
# analyse_train.drop(['YearRemodAdd'], axis = 1, inplace = True)
analyse_train.drop(['Utilities'], axis = 1, inplace = True)




# Add total living area combining above ground area and basement area
analyse_train['_TotLivArea'] = analyse_train['TotalBsmtSF'] + analyse_train['GrLivArea'] 



# Combine KitchenQual, OverallQual, ExterQual, BsmtQual, GarageQual, OverallCond, ExterCond, BsmtCond, GarageCond
analyse_train['_OverallQual_Cond'] = analyse_train['OverallCond'] * analyse_train['OverallQual']
analyse_train['_BsmtQual_Cond'] = analyse_train['BsmtQual'] * analyse_train['BsmtCond']
analyse_train['_ExterQual_Cond'] = analyse_train['ExterQual'] * analyse_train['ExterCond']
analyse_train['_GarageQual_Cond'] = analyse_train['GarageQual'] * analyse_train['GarageCond']

# Drop TotalBsmtSF and GrLivArea


# Scale GarageCars and GarageArea before combining them into one variable
analyse_train['GarageCars'] = abs(analyse_train['GarageCars'].transform(lambda x: (x - x.mean()) / x.std()))
analyse_train['GarageArea'] = abs(analyse_train['GarageArea'].transform(lambda x: (x - x.mean()) / x.std()))

analyse_train['_GarageCars_Area'] = analyse_train['GarageCars'] + analyse_train['GarageArea']



# Outdoor area  = WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea
analyse_train['EnclosedPorch'] = train_and_test['EnclosedPorch']
analyse_train['_Outdoor_Area'] = analyse_train['WoodDeckSF'] + analyse_train['OpenPorchSF'] + analyse_train['EnclosedPorch'] + analyse_train['3SsnPorch'] + analyse_train['ScreenPorch'] + analyse_train['PoolArea']


# Combine Fireplaces and FireplaceQu
analyse_train['_Fireplaces_Qu'] = analyse_train['Fireplaces'] + analyse_train['FireplaceQu']


# Add all rooms in the property
analyse_train['_Total_Rooms'] = analyse_train['TotRmsAbvGrd'] + analyse_train['FullBath'] + analyse_train['HalfBath'] + analyse_train['BsmtFullBath'] + analyse_train['BsmtHalfBath']






# =============================================================================
# # GrLivArea is sum of these three variables
# analyse_train.drop(['1stFlrSF'], axis = 1, inplace = True)
# analyse_train.drop(['2ndFlrSF'], axis = 1, inplace = True)
# analyse_train.drop(['LowQualFinSF'], axis = 1, inplace = True)
# # TotalBsmtSF is sum of these three variables
# analyse_train.drop(['BsmtFinSF1'], axis = 1, inplace = True)
# analyse_train.drop(['BsmtFinSF2'], axis = 1, inplace = True)
# analyse_train.drop(['BsmtUnfSF'], axis = 1, inplace = True)
# # Drop TotalBsmtSF and GrLivArea
# analyse_train.drop(['TotalBsmtSF'], axis = 1, inplace = True)
# analyse_train.drop(['GrLivArea'], axis = 1, inplace = True)
# analyse_train.drop(['OverallCond'], axis = 1, inplace = True)
# analyse_train.drop(['OverallQual'], axis = 1, inplace = True)
# analyse_train.drop(['BsmtQual'], axis = 1, inplace = True)
# analyse_train.drop(['BsmtCond'], axis = 1, inplace = True)
# analyse_train.drop(['ExterQual'], axis = 1, inplace = True)
# analyse_train.drop(['ExterCond'], axis = 1, inplace = True)
# analyse_train.drop(['GarageQual'], axis = 1, inplace = True)
# analyse_train.drop(['GarageCond'], axis = 1, inplace = True)
# analyse_train.drop(['GarageCars'], axis = 1, inplace = True)
# analyse_train.drop(['GarageArea'], axis = 1, inplace = True)
# analyse_train.drop(['WoodDeckSF'], axis = 1, inplace = True)
# analyse_train.drop(['OpenPorchSF'], axis = 1, inplace = True)
# analyse_train.drop(['EnclosedPorch'], axis = 1, inplace = True)
# analyse_train.drop(['3SsnPorch'], axis = 1, inplace = True)
# analyse_train.drop(['ScreenPorch'], axis = 1, inplace = True)
# analyse_train.drop(['PoolArea'], axis = 1, inplace = True)
# analyse_train.drop(['Fireplaces'], axis = 1, inplace = True)
# analyse_train.drop(['FireplaceQu'], axis = 1, inplace = True)
# analyse_train.drop(['TotRmsAbvGrd'], axis = 1, inplace = True)
# analyse_train.drop(['FullBath'], axis = 1, inplace = True)
# analyse_train.drop(['HalfBath'], axis = 1, inplace = True)
# analyse_train.drop(['BsmtHalfBath'], axis = 1, inplace = True)
# analyse_train.drop(['BsmtFullBath'], axis = 1, inplace = True)
# =============================================================================

# Land shape - not needed?

#############################################################################################################################
# Defininig own class for encoding multiple columns
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

#############################################################################################################################
# Labelencoding Vs Get Dummies
# temp = MultiColumnLabelEncoder(columns = ['SaleCondition']).fit_transform(analyse_train)



#############################################################################################################################
# Analysing the distribution

# SALE PRICE
analyse_train.reset_index(drop = True, inplace = True)
# analyse_train_y = np.log(analyse_train_y)

# Total Livable Area
analyse_train['_TotLivArea'] = np.log(analyse_train['_TotLivArea'])


# Total MasVnrArea
analyse_train['HasMasVnrArea'] = pd.Series(len(analyse_train['MasVnrArea']), index=analyse_train.index)
analyse_train['HasMasVnrArea'] = 0 
analyse_train.loc[analyse_train['MasVnrArea']>0,'HasMasVnrArea'] = 1
analyse_train.loc[analyse_train['HasMasVnrArea']==1,'MasVnrArea'] = np.log(analyse_train['MasVnrArea'])
analyse_train.drop(['HasMasVnrArea'], axis = 1, inplace = True)

# Total _Outdoor_Area
analyse_train['_HasOutdoor_Area'] = pd.Series(len(analyse_train['_Outdoor_Area']), index=analyse_train.index)
analyse_train['_HasOutdoor_Area'] = 0 
analyse_train.loc[analyse_train['_Outdoor_Area']>0,['_HasOutdoor_Area']] = 1
analyse_train.loc[analyse_train['_HasOutdoor_Area']==1,'_Outdoor_Area'] = np.log(analyse_train['_Outdoor_Area'])
analyse_train.drop(['_HasOutdoor_Area'], axis = 1, inplace = True)

# analyse_train.drop(['const'], axis = 1, inplace = True)


#############################################################################################################################
#
numeric_feats = analyse_train.dtypes[analyse_train.dtypes == "uint8"].index
skewed_feats = analyse_train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(30)

skewness = skewness[abs(skewness) > 0.75]

# skewed_features = skewness[abs(skewness) > 0.75].index
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    analyse_train[feat] = boxcox1p(analyse_train[feat], lam)


# Split train and test of transformed data
analyse_test = analyse_train.iloc[1457:,:]
analyse_train = analyse_train.iloc[:1457,:]
analyse_test.reset_index(drop = True, inplace = True)

#############################################################################################################################
#
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


#Validation function
n_folds = 10

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(analyse_train.values)
    rmse= np.sqrt(-cross_val_score(model, analyse_train.values, analyse_train_y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


#############################################################################################################################
#
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 


averaged_models = AveragingModels(models = (ENet,  lasso, model_xgb, GBoost, model_lgb, KRR))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

averaged_models.fit(analyse_train.values, analyse_train_y)
averaged_train_pred = averaged_models.predict(analyse_train.values)
averaged_test_pred_exp = np.exp(averaged_models.predict(analyse_test.values))
print(rmsle(analyse_train_y, averaged_train_pred))

np.savetxt("tenth_submission.csv",ensemble,delimiter=",")



class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(base_models = (ENet, model_xgb, GBoost, model_lgb, KRR),
                                                 meta_model = lasso)

analyse_train_y = np.array(analyse_train_y)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


stacked_averaged_models.fit(analyse_train.values, analyse_train_y)
stacked_train_pred = stacked_averaged_models.predict(analyse_train.values)
stacked_pred = np.exp(stacked_averaged_models.predict(analyse_test.values))
print(rmsle(analyse_train_y, stacked_train_pred))


lasso.fit(analyse_train, analyse_train_y)
lasso_train_pred = lasso.predict(analyse_train)
lasso_test_pred_exp = np.exp(lasso.predict(analyse_test))
print(rmsle(analyse_train_y, lasso_train_pred))


ENet.fit(analyse_train, analyse_train_y)
ENet_train_pred = ENet.predict(analyse_train)
ENet_test_pred_exp = np.exp(ENet.predict(analyse_test))
print(rmsle(analyse_train_y, ENet_train_pred))


model_xgb.fit(analyse_train, analyse_train_y)
model_xgb_train_pred = model_xgb.predict(analyse_train)
model_xgb_test_pred_exp = np.exp(model_xgb.predict(analyse_test))
print(rmsle(analyse_train_y, model_xgb_train_pred))


GBoost.fit(analyse_train, analyse_train_y)
GBoost_train_pred = GBoost.predict(analyse_train)
GBoost_test_pred_exp = np.exp(GBoost.predict(analyse_test))
print(rmsle(analyse_train_y, GBoost_train_pred))

model_lgb.fit(analyse_train, analyse_train_y)
model_lgb_train_pred = model_lgb.predict(analyse_train)
model_lgb_test_pred_exp = np.exp(model_lgb.predict(analyse_test))
print(rmsle(analyse_train_y, model_lgb_train_pred))


KRR.fit(analyse_train, analyse_train_y)
KRR_train_pred = KRR.predict(analyse_train)
KRR_test_pred_exp = np.exp(KRR.predict(analyse_test))
print(rmsle(analyse_train_y, KRR_train_pred))




ensemble = (stacked_pred * 0.7) + (model_lgb_test_pred_exp*0.15) + (GBoost_test_pred_exp*0.15) 


#############################################################################################################################
# Regression models


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 3000, random_state = 41)
regressor.fit(analyse_train, analyse_train_y)

y_pred = regressor.predict(analyse_train)

y_pred_normal = np.exp(y_pred)

np.savetxt("second_submission.csv",y_pred_normal,delimiter=",")

y_first_submission = pd.read_csv('House_prices_submission.csv')
y_first_submission.drop(['Id'], axis = 1, inplace = True)

import sklearn.metrics as mt
msle = mt.mean_squared_log_error(np.array(y_first_submission), np.array(y_pred_normal))


# missing_cols = set( analyse_train.columns ) - set( analyse_train_corr.columns )


#############################################################################################################################
# SCRATCH-----SCRATCH-----SCRATCH-----SCRATCH-----SCRATCH-----SCRATCH-----SCRATCH-----SCRATCH-----SCRATCH-----SCRATCH-----
# try adding 


sns.distplot(np.log(analyse_train), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log(analyse_train), plot=plt) 

sns.distplot(np.log(analyse_train[analyse_train['_DeflatedSalePrice']>0]['_DeflatedSalePrice']), fit=norm);


res = stats.probplot(np.log(analyse_train[analyse_train['_DeflatedSalePrice']>0]['_DeflatedSalePrice']), plot=plt)


group_columns_by_datatypes = analyse_train_y.columns.to_series().groupby(analyse_train_y.dtypes).groups
group_columns_by_datatypes


group_columns_by_datatypes = analyse_train_y.groupby(analyse_train_y.dtypes).groups
group_columns_by_datatypes

'MSSubClass', 'MSZoning', 'Alley', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
        'GarageYrBlt', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType',
        'SaleCondition'])



%%capture
#gather features
features = "+".join(analyse_train.columns)


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(analyse_train.values, i) for i in range(analyse_train.shape[1])]
vif["features"] = analyse_train.columns

vif0 = variance_inflation_factor(analyse_train.values, 1)


analyse_train.to_csv('analyse_train.csv', sep=',', index = False)

#############################################################################################################################
#
var = 'OverallQual'
data = pd.concat([train_and_test['SalePrice'], train_and_test[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

######################################

final_test = train_and_test.iloc[1457:,:]
final_train = train_and_test.iloc[:1457,:]

final_train_y = final_train['SalePrice']

final_train.drop(['SalePrice'], axis = 1, inplace = True)
final_test.drop(['SalePrice'], axis = 1, inplace = True)

final_train.drop(['Id'], axis = 1, inplace = True)


final_train = pd.get_dummies(final_train,columns = ['MSSubClass', 'MSZoning', 'Alley', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
        'GarageYrBlt', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType',
        'SaleCondition'])

final_train.drop(['CentralAir'], axis = 1, inplace = True)
final_train.drop(['GarageType'], axis = 1, inplace = True)
final_train.drop(['EnclosedPorch'], axis = 1, inplace = True)
final_train.drop(['LowQualFinSF'], axis = 1, inplace = True)
final_train.drop(['YearRemodAdd'], axis = 1, inplace = True)



final_train_opt = np.append(arr = np.ones((1457,1)).astype(int) , values = final_train, axis = 1)

# final_train_opt = final_train[:,:]
regressor_OLS = sm.OLS(endog = np.asarray(final_train_y), exog = final_train_opt).fit()

regressor_OLS.summary()



pca = PCA(n_components = None)
final_train = pca.fit_transform(final_train)
explained_variance = pca.explained_variance_ratio_

lda = LDA(n_components = None)
final_train = lda.fit_transform(final_train, final_train_y)
lda_explained_variance = lda.explained_variance_ratio_




# Plots
k = 11 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(final_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

var = 'MSSubClass'
data = pd.concat([train_and_test['SalePrice'], train_and_test[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, 
            square=True, annot=True);

            
k = 25 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()



###########################################
analyse_train_corr = analyse_train.copy()
analyse_train_corr['SalePrice'] = analyse_train_y
corrmat = analyse_train_corr.corr()


            
k = 27 #number of variables for heatmap
cols = corrmat.nlargest(k, 'OverallQual')['OverallQual'].index
cm = np.corrcoef(analyse_train_corr[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
###################################################


exog = np.array(train[cols]).transpose()
vif0 = oi.variance_inflation_factor(exog, 0)
















#############################################################################################################################
# Calculating VIF







































































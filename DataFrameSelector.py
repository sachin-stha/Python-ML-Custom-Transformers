import sklearn as sk

class DataFrameSelector(sk.base.BaseEstimator, sk.base.TransformerMixin):
    def __init__(self, attr = []):
        self.attr = attr
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # if user doesn't pass attributes, it will select all the columns present in the dataframe
        if self.attr == []:
            return X.values 
        else:
            return X[self.attr].values

# Method of using this Transformer
# It is same as using other Pre-build python Transfromer
# It's Object accepts parameter which is attributes to be Selected
#  For Example :
      # selector = DataFrameSelector([a, b, c, d])
      # In this example, Column 'a', 'b', 'c' and 'd' is selected.
      # It can only select attribute which the Dataframe contains.
# After methods are same as other Pre-build Transformer
      # for fit and transform, data must be passed
      # For example : selector.fit_transform(data) // Continued from previous example

# --- It is mostly useful for Pipeline
# for example - 
  # pipeline = Pipeline([
  #         ('selector', DataFrameSelector()),
  #         ('imputer', SimpleImputer(strategy='median')),
  #         ('Std_scaler', StandardScaler())
  #     ])
  # In this examples, this transformer passes selected data to another transformer, which is SimpleImputer in this case.

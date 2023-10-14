import sklearn as sk
import pandas as pd
import numpy as np

class CombinedAttrAdder(sk.base.BaseEstimator, sk.base.TransformerMixin):
    def __init__(self, attr):
        self.attr = attr 
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        new_attr = {} 
        X_temp = X.copy()
        
        for attrs in self.attr:
            new_attr[''.join([attrs[0].lower(), '_per_', attrs[1].lower()])] = X[attrs[0]] / X[attrs[1]]
            
        for new_attrs_data in new_attr:
            X_temp = pd.DataFrame(np.c_[X_temp, pd.DataFrame(new_attr[new_attrs_data])]) # joins input data with new combined attribute data
        
        X_temp.columns = np.concatenate((np.array(X.columns), np.array([a for a in new_attr])), axis=0) # changes the name of columns to its originals
        return X_temp

# Method of using this Transformer
# It is same as using other Pre-build python Transfromer
# It's Object accepts parameter which is attributes to be Combined
#  For Example :
      # attr_adder = CombinedAttrAdder([[a, b], [d, e]])
      # In this example, 'a' column is divided with 'b' column and 'd' column is divided with 'e' column
      # Only two attributes can be combined, but as many combination of attributes can be passed to transformers as you want
# After methods are same as other Pre-build Transformer
      # for fit and transform, data must be passed
      # For example : attr_adder.fit_transform(data) // Continued from previous example

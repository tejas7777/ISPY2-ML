from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np



class LassoSelector:

    data: pd.DataFrame = None
    target = None

    def __init__(self,data,target):
        self.data = data
        self.target = target

    def select(self)->list[str]:
        #Selects dependent variable
        y = self.data[self.target]
        self.data.drop(columns=[self.target])
        
        #Selects independent variables
        X = self.data.drop(columns=[self.target])

        #Feature names
        features = X.columns

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Create a LASSO model
        alpha = 0.01  # Adjust the regularization strength
        lasso_model = Lasso(random_state=42)

        # Fit the model
        lasso_model.fit(X_train_scaled, y_train)

        # Check selected features
        selected_feature_indices = np.where(lasso_model.coef_ != 0)[0]
        selected_features = features[selected_feature_indices]
        
        return selected_features



    

        
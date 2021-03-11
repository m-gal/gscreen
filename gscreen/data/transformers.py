""" Contains the custom data transformers used in the project

    Created on Dec 08 2020
    @author: mikhail.galkin
"""

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class TransportTypeTransformer(TransformerMixin, BaseEstimator):
    """Make one hot encoding for transport_type column
    a general class for creating a machine learning step in the ml pipeline
    """

    def __init__(self):
        """constructor"""
        return None

    # Return self, nothing else to do here
    def fit(self, X, y=None, **fit_params):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        transport_dummies = pd.get_dummies(
            X["transport_type"],
            prefix="transport_type",
        )
        self.transport_columns = transport_dummies.columns
        return self

    # Custom date transform metohd
    def transform(self, X, y=None, **transform_params):
        """
        an abstract method that is used to transform according to what happend
        in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        transport_dummies = pd.get_dummies(
            X["transport_type"],
            prefix="transport_type",
        )
        transport_dummies = transport_dummies.reindex(
            columns=self.transport_columns,
            fill_value=0,
        )
        X = pd.concat([X, transport_dummies], axis=1)
        X.drop("transport_type", axis=1, inplace=True)

        return X


class PickupDateTransformer(TransformerMixin, BaseEstimator):
    """Extracts features from datetime column
    a general class for creating a machine learning step in the ml pipeline
    """

    def __init__(self):
        """constructor"""
        return None

    # Return self, nothing else to do here
    def fit(self, X, y=None, **fit_params):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer
        """
        return self

    # Custom date transform metohd
    def transform(self, X, y=None, **transform_params):
        """
        an abstract method that is used to transform according to what happend
        in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        """ Derive Month, Week, Day, Weekday and Hour from pickup_date"""
        # Just in case convert to datetime
        X["pickup_date"] = pd.to_datetime(
            X["pickup_date"],
            format="%Y-%m-%d %H:%M:%S",
        )
        # Parse datetime
        # X["pickup_quarter"] = X["pickup_date"].dt.quarter
        # X["pickup_quarterstart"] = X["pickup_date"].dt.is_quarter_start.astype(int)
        # X["pickup_quarterend"] = X["pickup_date"].dt.is_quarter_end.astype(int)
        # X["pickup_month"] = X["pickup_date"].dt.month
        # X["pickup_monthstart"] = X["pickup_date"].dt.is_month_start.astype(int)
        # X["pickup_monthend"] = X["pickup_date"].dt.is_month_end.astype(int)
        X["pickup_week"] = X["pickup_date"].dt.isocalendar().week
        # X["pickup_dayofmonth"] = X["pickup_date"].dt.day
        # X["pickup_dayofweek"] = X["pickup_date"].dt.dayofweek
        # X["pickup_dayofyear"] = X["pickup_date"].dt.dayofyear
        X["pickup_hour"] = X["pickup_date"].dt.hour

        # X["pickup_quarter_sin"] = np.sin(2 * np.pi * X["pickup_date"].dt.quarter/4)
        # X["pickup_month_sin"] = np.sin(2 * np.pi * X["pickup_date"].dt.month/12)
        # X["pickup_week_sin"] = np.sin(2 * np.pi * X["pickup_date"].dt.isocalendar().week/52)
        # X["pickup_dayofmonth_sin"] = np.sin(2 * np.pi * X["pickup_date"].dt.day/31)
        # X["pickup_dayofweek_sin"] = np.sin(2 * np.pi * X["pickup_date"].dt.dayofweek/6)
        X["pickup_dayofyear_sin"] = np.sin(2 * np.pi * X["pickup_date"].dt.dayofyear / 365)
        # X["pickup_hour_sin"] = np.sin(2 * np.pi * X["pickup_date"].dt.hour/23)

        X.drop(["pickup_date"], axis=1, inplace=True)

        return X


class InsideStateTransformer(TransformerMixin, BaseEstimator):
    """Extracts features from KMA columns
    a general class for creating a machine learning step in the ml pipeline
    """

    def __init__(self):
        """constructor"""
        return None

    # Return self, nothing else to do here
    def fit(self, X, y=None, **fit_params):
        return self

    # Custom date transform metohd
    def transform(self, X, y=None, **transform_params):
        """ Make feature if moving is in one state """
        ## TODO: SettingWithCopyWarning raises. It is problem of Pandas.
        ## A value is trying to be set on a copy of a slice from a DataFrame.
        ## Try using .loc[row_indexer,col_indexer] = value instead
        orig = pd.Series(X.loc[:, "origin_kma"])
        dest = pd.Series(X.loc[:, "destination_kma"])
        X["instate"] = np.where(orig.str.slice(0, 2) == dest.str.slice(0, 2), 1, 0)
        return X

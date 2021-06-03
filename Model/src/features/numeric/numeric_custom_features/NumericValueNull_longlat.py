import pandas as pd
from math import cos, sin
from src.features.FeatureETLEngineering import FeatureETLEngineering


class NumericValueNull_longlat(FeatureETLEngineering):
    def __init__(self, col_series):
        """Instantiate categorical feature."""
        super().__init__(col_series)

    def run_etl(self):
        """Run ETL of the column."""
        self._add_is_null_column_df()
        self._propertyurl_to_address()
        self._address_to_coordinates()
        self._LatLong_to_sphere_coor()
        self._build_feature_etl_df()

    def _propertyurl_to_address(self):
        """Turn PropertyUrl into Address"""
        self.col_series = self.col_series.str.split('_FL_', expand = True)[0]
        self.col_series = self.col_series.str.split('/', expand = True)
        self.col_series = self.col_series[4].str.replace('-', ' ')
        self.col_series = self.col_series.str.replace('_', ' ')

    def _address_to_coordinates(self):
        """Turn Address into Longitutde and Long Coordinates"""

        Use self.col_series as input..

        self.df_value['Longitutde'] = 
        self.df_value['Latitude'] = 

    def _LatLong_to_sphere_coor(self):
        """LatLong to 3-D Spherical Coordinates via Sin and Cos operations"""
        self.df_value['latlong_x'] = cos(self.df_value['Latitude']) * cos(self.df_value['Longitude'])
        self.df_value['latlong_y'] = cos(self.df_value['Latitude']) * sin(self.df_value['Longitude']) 
        self.df_value['latlong_z'] = sin(self.df_value['Latitude']) 

    
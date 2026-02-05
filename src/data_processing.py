import pandas as pd
import numpy as np
import json
from typing import Dict, List
import os

class DataProcessor:
    def __init__(self, column_path: str = 'models/columns.json'):
        with open(column_path, 'r') as f:
            self.columns_info = json.load(f)
            
            self.feature_columns = self.columns_info['feature_columns']
            self.top_locations = self.columns_info['top_locations']
    
    def preprocess_input(self, input_data: Dict) -> pd.DataFrame:
        
        # base features
        processed = {
            'total_sqft': float(input_data['total_sqft']),
            'bath': float(input_data['bath']),
            'bhk': float(input_data['bhk']),
            'balcony': float(input_data['balcony'])
        }
        
        # location handling
        location = input_data['location']
        location_category = location if location in self.top_locations else 'other'
        processed['location_category'] = location_category
        
        # modifying features
        processed['price_per_sqft'] = 0
        processed['bath_per_bhk'] = processed['bath'] / processed['bhk']
        processed['is_studio'] = int(processed['bhk'] == 1 and processed['total_sqft'] < 500)
        
        # categorical data handling
        processed['area_type'] = input_data.get('area_type', 'Super built-up Area')
        availability = input_data.get('availability', 'Ready To Move')
        availability_category = availability if availability == 'Ready To Move' else 'Not Ready To Move'
        processed['availability_category'] = availability_category
        
        df = pd.DataFrame([processed])
        
        # one hot encoding
        df = pd.get_dummies(df, columns=["area_type", "availability_category", "location_category"], drop_first = True)
        
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_columns]
        
        return df
    
    def get_available_locations(self) -> List[str]:
        return self.top_locations + ['other']
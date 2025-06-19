# data_loader.py

import pandas as pd

def load_data():
    # Load all three datasets
    df_bike = pd.read_csv("data/2-wheeler-EV-bikewale.csv")
    df_carwale = pd.read_csv("data/4-wheeler-EV-carwale.csv")
    df_cardekho = pd.read_csv("data/4-wheeler-EV-cardekho.csv")

    # Standardize column names
    df_bike = df_bike.rename(columns={
        "rating": "Rating",
        "Review": "Review",
        "Model_Name": "Model_Name"
    })

    df_carwale = df_carwale.rename(columns={
        "rating": "Rating",
        "Review": "Review",
        "Model_Name": "Model_Name"
    })

    df_cardekho = df_cardekho.rename(columns={
        "Rating": "Rating",
        "Review": "Review",
        "Model_Name": "Model_Name",
        "Attributes Mentioned": "Attributes"
    })

    df_bike["Vehicle_Type"] = "2W"
    df_carwale["Vehicle_Type"] = "4W"
    df_cardekho["Vehicle_Type"] = "4W"

    # Add missing attribute columns for consistency
    bike_attributes = ['Visual Appeal', 'Reliability', 'Performance', 'Service Experience', 'Extra Features', 'Comfort', 'Maintenance cost', 'Value for Money']
    for col in bike_attributes:
        if col not in df_bike.columns:
            df_bike[col] = None

    car_attributes = ['Exterior', 'Comfort', 'Performance', 'Fuel Economy', 'Value for Money', 'Condition']
    for col in car_attributes:
        if col not in df_carwale.columns:
            df_carwale[col] = None

    # We'll handle `Attributes` from cardekho in a separate logic for Tab 3

    return df_bike, df_carwale, df_cardekho

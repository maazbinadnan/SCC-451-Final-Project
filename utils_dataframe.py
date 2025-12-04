import pandas as pd
import os
def create_dataframe(filepath: str) -> pd.DataFrame:
    names = [
        "Temp(min)", "Temp(Max)", "Temp(Mean)",
        "Rel_Humid(Min)", "Rel_Humid(Max)", "Rel_Humid(Mean)",
        "Sea_Level_Pressure(Min)", "Sea_Level_Pressure(Max)", "Sea_Level_Pressure(Mean)",
        "precipitation_total", "snowfall_amount", "sunshine_duration",
        "wind_gust(min)", "wind_gust(max)", "wind_gust(mean)",
        "wind_speed(min)", "wind_speed(max)", "wind_speed(mean)"
    ]
    lowercase_names = [name.lower() for name in names]
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    
    # Load data
    df = pd.read_csv(filepath, names=lowercase_names, header=None)
        
    return df
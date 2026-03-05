import pandas as pd
import json

with open("keyword_list.json", "r") as klist:
    keyword_list = json.load(klist)
    
keyword_df = pd.DataFrame(keyword_list)
keyword_df.to_csv("keyword_list.csv")
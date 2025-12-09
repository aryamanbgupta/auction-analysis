import pandas as pd
df = pd.read_csv('data/ipl_2026_auction_list.csv')
print("Original Columns:", df.columns.tolist())
df.columns = df.columns.str.replace(r'<[^>]+>', '', regex=True).str.strip()
print("Cleaned Columns:", df.columns.tolist())

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import difflib

def normalize_name(name):
    if pd.isna(name): return ""
    return str(name).strip().lower()

class ValuationEngine:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.models = {}
        self.projections = {}
        self.metadata = None
        self.price_model = None
        self.role_columns = []
        
    def load_data(self):
        print("Loading data for Valuation Engine...")
        
        # 1. Load Metadata
        self.metadata = pd.read_csv(self.project_root / 'data' / 'player_metadata.csv')
        self.metadata['name_norm'] = self.metadata['player_name'].apply(normalize_name)
        
        # 2. Load Projections (2026)
        proj_dir = self.project_root / 'results' / 'WARprojections'
        
        # Load all 3 models for Batters and Bowlers
        dfs = []
        for role in ['batter', 'bowler']:
            name_col = 'batter_name' if role == 'batter' else 'bowler_name'
            
            # Marcel (usually has player_name)
            try:
                m = pd.read_csv(proj_dir / 'marcel' / f'{role}_projections_2026.csv')
                if 'player_name' in m.columns:
                    m = m[['player_name', 'projected_war_2026']]
                else:
                    m = m[[name_col, 'projected_war_2026']].rename(columns={name_col: 'player_name'})
                m = m.rename(columns={'projected_war_2026': 'marcel'})
            except Exception as e:
                print(f"Error loading Marcel {role}: {e}")
                m = pd.DataFrame(columns=['player_name', 'marcel'])

            # IPL ML
            try:
                i = pd.read_csv(proj_dir / f'{role}_projections_2026.csv')
                i = i[[name_col, 'projected_war_2026']].rename(columns={name_col: 'player_name', 'projected_war_2026': 'ipl_ml'})
            except Exception as e:
                print(f"Error loading IPL ML {role}: {e}")
                i = pd.DataFrame(columns=['player_name', 'ipl_ml'])
            
            # Global ML
            try:
                g = pd.read_csv(proj_dir / f'{role}_projections_2026_global.csv')
                g = g[[name_col, 'projected_war_2026']].rename(columns={name_col: 'player_name', 'projected_war_2026': 'global_ml'})
            except Exception as e:
                print(f"Error loading Global ML {role}: {e}")
                g = pd.DataFrame(columns=['player_name', 'global_ml'])
            
            # Merge
            merged = m.merge(i, on='player_name', how='outer').merge(g, on='player_name', how='outer')
            dfs.append(merged)
            
        self.projections = pd.concat(dfs, ignore_index=True)
        # Average Projection
        self.projections['war_2026'] = self.projections[['marcel', 'ipl_ml', 'global_ml']].mean(axis=1)
        self.projections['name_norm'] = self.projections['player_name'].apply(normalize_name)
        
        print(f"Loaded projections for {len(self.projections)} players.")
        
    def train_price_model(self):
        print("Training Price Model (VOMAM-style)...")
        # Load 2025 Price vs WAR data
        price_data = pd.read_csv(self.project_root / 'results' / '11_war_vs_price' / 'war_vs_price_full.csv')
        
        # Merge metadata for Role/Overseas
        price_data = price_data.merge(self.metadata[['player_name', 'country', 'role_category']], 
                                      left_on='cricwar_name', right_on='player_name', how='left')
        
        price_data['is_overseas'] = price_data['country'].apply(lambda x: 0 if str(x).lower() in ['india', 'ind'] else 1)
        price_data['role_category'] = price_data['role_category'].fillna('Unknown')
        
        # Features
        df_encoded = pd.get_dummies(price_data, columns=['role_category'], prefix='role', drop_first=False)
        self.role_columns = [c for c in df_encoded.columns if c.startswith('role_')]
        
        features = ['total_WAR', 'is_overseas'] + self.role_columns
        target = 'price_cr'
        
        X = df_encoded[features].fillna(0)
        y = df_encoded[target].fillna(0)
        
        self.price_model = LinearRegression()
        self.price_model.fit(X, y)
        print(f"Price Model Trained. R2: {self.price_model.score(X, y):.3f}")
        
    def predict_price(self, war, is_overseas, role):
        # Create input df
        input_data = pd.DataFrame({'total_WAR': [war], 'is_overseas': [int(is_overseas)]})
        for col in self.role_columns:
            role_name = col.replace('role_', '')
            input_data[col] = 1 if role == role_name else 0
        
        # Ensure all columns match training
        # (LinearRegression in sklearn depends on column order if numpy array, but we pass DF? No, sklearn converts to array)
        # We must ensure column order matches X.columns from training
        # Re-construct feature list
        features = ['total_WAR', 'is_overseas'] + self.role_columns
        return self.price_model.predict(input_data[features])[0]

    def process_targets(self):
        target_dir = self.project_root / 'results' / 'analysis' / 'strategic'
        output_dir = target_dir / 'valued'
        output_dir.mkdir(exist_ok=True)
        
        target_files = list(target_dir.glob('target_*.csv'))
        
        for file in target_files:
            print(f"\nProcessing {file.name}...")
            df = pd.read_csv(file)
            
            # Identify Name column (usually index or first column if saved without index=False? 
            # strategic_analysis.py saves with index=True (default) so Player Name is index or first col)
            # Let's check the CSV structure. usually: Player,SR,RAA...
            if 'Player' not in df.columns:
                # Assume first column is player name
                df = df.rename(columns={df.columns[0]: 'Player'})
            
            results = []
            for _, row in df.iterrows():
                name = row['Player']
                name_norm = normalize_name(name)
                
                # 1. Get Projection
                proj_row = self.projections[self.projections['name_norm'] == name_norm]
                if proj_row.empty:
                    # Fuzzy match?
                    # For now, skip or use 0
                    war_proj = 0.0
                    print(f"  Warning: No projection for {name}")
                else:
                    war_proj = proj_row['war_2026'].values[0]
                
                # 2. Get Metadata (Role, Country)
                meta_row = self.metadata[self.metadata['name_norm'] == name_norm]
                if meta_row.empty:
                    role = 'Unknown'
                    country = 'Unknown'
                else:
                    role = meta_row['role_category'].values[0]
                    country = meta_row['country'].values[0]
                
                is_overseas = 0 if str(country).lower() in ['india', 'ind'] else 1
                
                # 3. Predict Price
                # If WAR is NaN (e.g. new player), use 0 or skip
                if pd.isna(war_proj): war_proj = 0.0
                
                price_pred = self.predict_price(war_proj, is_overseas, role)
                price_pred = max(0.2, price_pred) # Minimum base price approx
                
                # Collect Result
                res = row.to_dict()
                res['Role'] = role
                res['Country'] = country
                res['WAR_2026_Proj'] = round(war_proj, 2)
                res['Est_Price_Cr'] = round(price_pred, 2)
                results.append(res)
            
            # Save Valued CSV
            if not results:
                print(f"  No results for {file.name}")
                continue
                
            res_df = pd.DataFrame(results)
            # Reorder columns
            cols = ['Player', 'Role', 'Country', 'WAR_2026_Proj', 'Est_Price_Cr'] + [c for c in res_df.columns if c not in ['Player', 'Role', 'Country', 'WAR_2026_Proj', 'Est_Price_Cr']]
            res_df = res_df[cols]
            
            out_file = output_dir / f"valued_{file.name}"
            res_df.to_csv(out_file, index=False)
            print(f"  Saved to {out_file}")
            print(res_df[['Player', 'WAR_2026_Proj', 'Est_Price_Cr']].head().to_string(index=False))

def main():
    root = Path(__file__).parent.parent.parent
    engine = ValuationEngine(root)
    engine.load_data()
    engine.train_price_model()
    engine.process_targets()

if __name__ == "__main__":
    main()

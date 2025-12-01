import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import difflib

def normalize_name(name):
    if pd.isna(name): return ""
    return str(name).strip().lower()

class ValuationEngine:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.projections = {}
        self.metadata = None
        self.vomam_model = None
        self.vope_params = None
        self.role_columns = []
        
    def load_data(self):
        print("Loading data for Valuation Engine...")
        
        # 1. Load Metadata
        self.metadata = pd.read_csv(self.project_root / 'data' / 'player_metadata.csv')
        self.metadata['name_norm'] = self.metadata['player_name'].apply(normalize_name)
        
        # 2. Load Historical Data (for Normalization)
        # We need to know the typical scale of WAR.
        # Let's load the 2022-2025 stats used in strategic analysis.
        try:
            hist_bat = pd.read_csv(self.project_root / 'results' / 'analysis' / 'auction_stats' / 'batters_2022_2025.csv')
            hist_bowl = pd.read_csv(self.project_root / 'results' / 'analysis' / 'auction_stats' / 'bowlers_2022_2025.csv')
            
            # These files have 'WAR' which is likely total over 4 years.
            # We need Per Season WAR to compare with 2026 Projections.
            # Approx: Total WAR / 4 (or number of seasons played).
            # Let's just take the 99th percentile of the Total WAR and divide by ~3 (avg seasons played) as a rough baseline?
            # Better: Load the raw WAR vs Price file which has 'total_WAR' (Per Season likely, or Total 2025).
            
            price_data = pd.read_csv(self.project_root / 'results' / '11_war_vs_price' / 'war_vs_price_full.csv')
            # This file was used for training. 'total_WAR' here is the target scale we want.
            target_99th = price_data['total_WAR'].quantile(0.99)
            print(f"Target Scale (Historical 99th %ile): {target_99th:.2f}")
            
        except Exception as e:
            print(f"Warning: Could not load historical data for normalization: {e}")
            target_99th = 2.0 # Fallback
            
        # 3. Load Projections (2026)
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
        
        # Sum duplicates (Player can be both batter and bowler)
        self.projections = self.projections.groupby('player_name').sum().reset_index()
        
        # Average Projection
        self.projections['war_2026_raw'] = self.projections[['marcel', 'ipl_ml', 'global_ml']].mean(axis=1)
        
        # --- Normalization ---
        proj_99th = self.projections['war_2026_raw'].quantile(0.99)
        print(f"Projected Scale (Raw 99th %ile): {proj_99th:.2f}")
        
        if proj_99th > 0:
            scaling_factor = target_99th / proj_99th
        else:
            scaling_factor = 1.0
            
        print(f"Applying Scaling Factor: {scaling_factor:.3f}")
        self.projections['war_2026'] = self.projections['war_2026_raw'] * scaling_factor
        
        self.projections['name_norm'] = self.projections['player_name'].apply(normalize_name)
        
        print(f"Loaded projections for {len(self.projections)} players.")
        
    def train_models(self):
        print("Training Price Models...")
        # Load 2025 Price vs WAR data
        price_data = pd.read_csv(self.project_root / 'results' / '11_war_vs_price' / 'war_vs_price_full.csv')
        
        # Merge metadata for Role/Overseas
        price_data = price_data.merge(self.metadata[['player_name', 'country', 'role_category']], 
                                      left_on='cricwar_name', right_on='player_name', how='left')
        
        price_data['is_overseas'] = price_data['country'].apply(lambda x: 0 if str(x).lower() in ['india', 'ind'] else 1)
        price_data['role_category'] = price_data['role_category'].fillna('Unknown')
        
        # --- 1. VOMAM (Linear Model) ---
        print("  Training VOMAM (Linear)...")
        df_encoded = pd.get_dummies(price_data, columns=['role_category'], prefix='role', drop_first=False)
        self.role_columns = [c for c in df_encoded.columns if c.startswith('role_')]
        
        features = ['total_WAR', 'is_overseas'] + self.role_columns
        target = 'price_cr'
        
        X = df_encoded[features].fillna(0)
        y = df_encoded[target].fillna(0)
        
        self.vomam_model = LinearRegression()
        self.vomam_model.fit(X, y)
        print(f"  VOMAM R2: {self.vomam_model.score(X, y):.3f}")
        
        # --- 2. VOPE (Power Law Model) ---
        print("  Training VOPE (Power Law)...")
        # Filter for positive WAR and Price to avoid log issues or bad fits
        vope_data = price_data[(price_data['total_WAR'] > 0) & (price_data['price_cr'] > 0)].copy()
        
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        try:
            popt, _ = curve_fit(power_law, vope_data['total_WAR'], vope_data['price_cr'], p0=[1, 1], maxfev=5000)
            self.vope_params = popt
            print(f"  VOPE Params: a={popt[0]:.3f}, b={popt[1]:.3f}")
            
            # Calculate R2 for VOPE
            y_pred = power_law(vope_data['total_WAR'], *popt)
            ss_res = np.sum((vope_data['price_cr'] - y_pred) ** 2)
            ss_tot = np.sum((vope_data['price_cr'] - np.mean(vope_data['price_cr'])) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            print(f"  VOPE R2: {r2:.3f}")
            
        except Exception as e:
            print(f"  Error training VOPE: {e}")
            self.vope_params = None
        
    def predict_price(self, war, is_overseas, role):
        # --- VOMAM Prediction ---
        input_data = pd.DataFrame({'total_WAR': [war], 'is_overseas': [int(is_overseas)]})
        for col in self.role_columns:
            role_name = col.replace('role_', '')
            input_data[col] = 1 if role == role_name else 0
            
        features = ['total_WAR', 'is_overseas'] + self.role_columns
        vomam_price = self.vomam_model.predict(input_data[features])[0]
        vomam_price = max(0.2, vomam_price)
        
        # --- VOPE Prediction ---
        vope_price = 0.2
        if self.vope_params is not None:
            if war > 0:
                vope_price = self.vope_params[0] * (war ** self.vope_params[1])
            else:
                vope_price = 0.2 # Base price for 0 or negative WAR
        
        # Ensure min price
        vope_price = max(0.2, vope_price)
        
        return round(vomam_price, 2), round(vope_price, 2)

    def process_targets(self):
        target_dir = self.project_root / 'results' / 'analysis' / 'strategic'
        output_dir = target_dir / 'valued'
        output_dir.mkdir(exist_ok=True)
        
        target_files = list(target_dir.glob('target_*.csv'))
        
        for file in target_files:
            print(f"\nProcessing {file.name}...")
            df = pd.read_csv(file)
            
            if 'Player' not in df.columns:
                # Assume first column is player name if not explicitly named
                # But check if 'Unnamed: 0' exists which might be the index
                if 'Unnamed: 0' in df.columns:
                     df = df.rename(columns={'Unnamed: 0': 'Player'})
                else:
                     df = df.rename(columns={df.columns[0]: 'Player'})
            
            results = []
            for _, row in df.iterrows():
                name = row['Player']
                name_norm = normalize_name(name)
                
                # 1. Get Projection
                proj_row = self.projections[self.projections['name_norm'] == name_norm]
                if proj_row.empty:
                    war_proj = 0.0
                    # print(f"  Warning: No projection for {name}")
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
                if pd.isna(war_proj): war_proj = 0.0
                
                # Scale WAR? 
                # The model was trained on 2025 WAR (Total Season). 
                # The projections are also Total Season (2026).
                # So no scaling needed if they are on the same scale.
                # NOTE: Check if 2025 WAR in training data is comparable to Projections.
                # Assuming yes for now.
                
                vomam, vope = self.predict_price(war_proj, is_overseas, role)
                
                # Collect Result
                res = row.to_dict()
                res['Role'] = role
                res['Country'] = country
                res['WAR_2026_Proj'] = round(war_proj, 2)
                res['Est_Price_VOMAM'] = vomam
                res['Est_Price_VOPE'] = vope
                res['Est_Price_Avg'] = round((vomam + vope) / 2, 2)
                results.append(res)
            
            # Save Valued CSV
            if not results:
                print(f"  No results for {file.name}")
                continue
                
            res_df = pd.DataFrame(results)
            
            # Reorder columns
            first_cols = ['Player', 'Role', 'Country', 'WAR_2026_Proj', 'Est_Price_VOMAM', 'Est_Price_VOPE', 'Est_Price_Avg']
            cols = first_cols + [c for c in res_df.columns if c not in first_cols]
            res_df = res_df[cols]
            
            out_file = output_dir / f"valued_{file.name}"
            res_df.to_csv(out_file, index=False)
            print(f"  Saved to {out_file}")
            print(res_df[['Player', 'WAR_2026_Proj', 'Est_Price_VOMAM', 'Est_Price_VOPE']].head().to_string(index=False))

def main():
    root = Path(__file__).parent.parent.parent
    engine = ValuationEngine(root)
    engine.load_data()
    engine.train_models()
    engine.process_targets()

if __name__ == "__main__":
    main()

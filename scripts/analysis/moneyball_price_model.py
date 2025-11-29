import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

class MoneyballPriceModel:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.model = None
        self.role_columns = []
        self.is_trained = False
        
    def load_and_train(self):
        print("Training Price Model...")
        data_path = self.project_root / 'results' / '11_war_vs_price' / 'war_vs_price_full.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Price data not found at {data_path}")
            
        df = pd.read_csv(data_path)
        
        # Load metadata for roles if not present
        if 'role_category' not in df.columns:
            meta_path = self.project_root / 'data' / 'player_metadata.csv'
            metadata = pd.read_csv(meta_path)
            # Merge
            df = df.merge(metadata[['player_name', 'country', 'role_category']], 
                          left_on='cricwar_name', right_on='player_name', how='left')
            
            df['is_overseas'] = df['country'].apply(lambda x: 0 if str(x).lower() in ['india', 'ind'] else 1)
            df['role_category'] = df['role_category'].fillna('Unknown')

        # Prepare Features
        # Target: price_cr
        # Features: total_WAR, is_overseas, role_dummies
        
        # One-hot encode roles
        df_encoded = pd.get_dummies(df, columns=['role_category'], prefix='role', drop_first=False)
        
        # Identify role columns
        self.role_columns = [c for c in df_encoded.columns if c.startswith('role_')]
        
        features = ['total_WAR', 'is_overseas'] + self.role_columns
        target = 'price_cr'
        
        X = df_encoded[features].fillna(0)
        y = df_encoded[target].fillna(0)
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.is_trained = True
        
        print(f"Model Trained. R2: {self.model.score(X, y):.3f}")
        
    def predict(self, war, is_overseas, role):
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call load_and_train() first.")
            
        # Create input dataframe
        input_data = pd.DataFrame({
            'total_WAR': [war],
            'is_overseas': [int(is_overseas)]
        })
        
        # Set role columns
        for col in self.role_columns:
            # Check if this column matches the input role
            # col format: role_Wicketkeeper
            role_name = col.replace('role_', '')
            input_data[col] = 1 if role == role_name else 0
            
        return self.model.predict(input_data)[0]

    def predict_batch(self, df):
        """
        Predict prices for a dataframe.
        df must have columns: 'WAR', 'is_overseas', 'role_category'
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
            
        # Prepare input
        df_encoded = pd.get_dummies(df, columns=['role_category'], prefix='role', drop_first=False)
        
        # Ensure all model columns exist
        for col in self.role_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        features = ['WAR', 'is_overseas'] + self.role_columns
        # Rename WAR to total_WAR if needed, but here we use the list directly
        # The model expects features in specific order? No, sklearn uses array, so order matters if we pass array.
        # But if we pass DataFrame with same column names?
        # Wait, sklearn strips column names. We MUST ensure column order matches training.
        
        # Let's reconstruct X with exact training columns
        # Training features: ['total_WAR', 'is_overseas', role_cols...]
        
        X = pd.DataFrame()
        X['total_WAR'] = df['WAR']
        X['is_overseas'] = df['is_overseas']
        
        # Map roles manually to ensure correctness
        for col in self.role_columns:
            role_name = col.replace('role_', '')
            X[col] = (df['role_category'] == role_name).astype(int)
            
        return self.model.predict(X)

if __name__ == "__main__":
    # Test
    root = Path(__file__).parent.parent.parent
    model = MoneyballPriceModel(root)
    model.load_and_train()
    
    # Test Prediction
    price = model.predict(war=2.0, is_overseas=1, role='Top-order Batter')
    print(f"Predicted Price for 2.0 WAR Overseas Top-order Batter: {price:.2f} Cr")

"""
XGBoost Price Prediction Model for IPL Auction.

Improves on VOPE/VOMAM by using comprehensive features:
- WAR projections (from V9 production model)
- Age at auction
- IPL experience (seasons played)
- Career WAR trajectory
- Role and overseas status
- Base price (anchor effect)

Training data: IPL 2025 prices (227 players)
Can be extended with historical auction data in: data/ipl_auction_historical.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class XGBoostPriceModel:
    """XGBoost-based price prediction for IPL auctions."""
    
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = Path(project_root)
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def load_training_data(self, auction_year=2025, include_historical=True):
        """Load price data and merge with player features."""
        print(f"Loading training data for {auction_year}...")
        
        # Load base data sources
        metadata = pd.read_csv(self.project_root / 'data' / 'player_metadata.csv')
        batter_hist = pd.read_csv(self.project_root / 'data' / 'batter_war_full_history.csv')
        bowler_hist = pd.read_csv(self.project_root / 'data' / 'bowler_war_full_history.csv')
        
        try:
            enriched = pd.read_csv(self.project_root / 'data' / 'all_players_enriched.csv')
        except:
            print("  Warning: Could not load enriched data for DOB")
            enriched = pd.DataFrame()
        
        all_data = []
        
        # --- Load 2025 data (current) ---
        war_price_2025 = pd.read_csv(self.project_root / 'results' / '11_war_vs_price' / 'war_vs_price_full.csv')
        df_2025 = war_price_2025.copy()
        df_2025['auction_year'] = 2025
        df_2025 = df_2025.merge(
            metadata[['player_name', 'country', 'role_category']],
            left_on='cricwar_name', right_on='player_name', how='left'
        )
        # Add default base price for 2025 (since we don't have it in the file)
        # Most sold players > 20L, but for training stability we set a baseline
        df_2025['base_price_lakh'] = 20.0
        
        df_2025 = self._add_history_features(df_2025, batter_hist, bowler_hist, 2025)
        if not enriched.empty:
            df_2025 = self._add_age_features(df_2025, enriched, 2025)
        df_2025 = df_2025[df_2025['match_type'] != 'none'].copy()
        print(f"  Loaded {len(df_2025)} players from 2025")
        all_data.append(df_2025)
        
        # --- Load historical data (2022) ---
        if include_historical:
            hist_file = self.project_root / 'data' / 'ipl_auction_historical.csv'
            if hist_file.exists():
                hist_df = pd.read_csv(hist_file)
                hist_df = hist_df[hist_df['year'] == 2022].copy()
                
                print(f"  Processing {len(hist_df)} historical records from 2022...")
                
                hist_features = []
                for _, row in hist_df.iterrows():
                    player_name = row['player_name']
                    
                    # Handle Unsold players
                    team = row.get('team', '')
                    if str(team).lower() == 'unsold' or pd.isna(row['price_cr']):
                        price = 0.0
                    else:
                        price = float(row['price_cr'])
                    
                    role = row.get('role', 'Unknown')
                    base_price_str = row.get('base_price', '20 Lakh')
                    base_price_lakh = self._parse_base_price(base_price_str)
                    
                    # Match to WAR history
                    bat_match = batter_hist[(batter_hist['batter_name'].str.contains(player_name.split()[-1], case=False, na=False)) & 
                                            (batter_hist['season'] == 2021)]
                    bowl_match = bowler_hist[(bowler_hist['bowler_name'].str.contains(player_name.split()[-1], case=False, na=False)) & 
                                             (bowler_hist['season'] == 2021)]
                    
                    total_war = 0
                    if len(bat_match) > 0:
                        total_war += bat_match['WAR'].sum()
                    if len(bowl_match) > 0:
                        total_war += bowl_match['WAR'].sum()
                    
                    # History
                    player_bat_hist = batter_hist[(batter_hist['batter_name'].str.contains(player_name.split()[-1], case=False, na=False)) & 
                                                  (batter_hist['season'] < 2022)]
                    player_bowl_hist = bowler_hist[(bowler_hist['bowler_name'].str.contains(player_name.split()[-1], case=False, na=False)) & 
                                                   (bowler_hist['season'] < 2022)]
                    
                    ipl_seasons = max(
                        player_bat_hist['season'].nunique() if len(player_bat_hist) > 0 else 0,
                        player_bowl_hist['season'].nunique() if len(player_bowl_hist) > 0 else 0
                    )
                    career_war = (player_bat_hist['WAR'].sum() if len(player_bat_hist) > 0 else 0) + \
                                 (player_bowl_hist['WAR'].sum() if len(player_bowl_hist) > 0 else 0)
                    
                    # Age
                    age = 28.0
                    if not enriched.empty:
                        name_parts = player_name.split()
                        for part in name_parts:
                            if len(part) > 2:
                                match = enriched[enriched['name'].str.contains(part, case=False, na=False)]
                                if len(match) > 0 and pd.notna(match.iloc[0]['dob']):
                                    dob = pd.to_datetime(match.iloc[0]['dob'], errors='coerce')
                                    if pd.notna(dob):
                                        age = (pd.Timestamp('2022-01-01') - dob).days / 365.25
                                        break
                    
                    # Overseas
                    is_overseas = 1
                    if any(x in player_name.lower() for x in ['sharma', 'pant', 'kohli', 'dhoni', 'iyer', 'pandya', 'yadum', 'singh', 'kumar']):
                        is_overseas = 0
                    
                    hist_features.append({
                        'cricwar_name': player_name,
                        'price_cr': price,
                        'total_WAR': total_war,
                        'auction_year': 2022,
                        'ipl_seasons': ipl_seasons,
                        'career_war': career_war,
                        'recent_war': total_war,
                        'war_consistency': 0,
                        'age': age,
                        'is_overseas': is_overseas,
                        'role_category': self._standardize_role(role),
                        'base_price_lakh': base_price_lakh
                    })
                
                df_2022 = pd.DataFrame(hist_features)
                print(f"  Processed {len(df_2022)} historical records (including unsold)")
                all_data.append(df_2022)

        
        # Combine all years
        combined = pd.concat(all_data, ignore_index=True)
        
        # Clean up
        combined['is_overseas'] = combined.apply(
            lambda x: 0 if str(x.get('country', '')).lower() in ['india', 'ind'] else x.get('is_overseas', 1),
            axis=1
        )
        combined['role_category'] = combined['role_category'].fillna('Unknown')
        
        print(f"  Final training set: {len(combined)} players with all features")
        return combined
    
    
    def _parse_base_price(self, price_str):
        """Convert base price string to Lakhs."""
        if not isinstance(price_str, str):
            return 20.0
        price_str = price_str.lower().strip()
        if 'cr' in price_str:
            return float(price_str.replace('cr', '').strip()) * 100
        elif 'lakh' in price_str:
            return float(price_str.replace('lakh', '').strip())
        elif 'draft' in price_str or 'retain' in price_str:
            return 200.0  # High base for retained
        return 20.0

    def _parse_price(self, price_str):
        """Convert price string to Crores."""
        if not isinstance(price_str, str):
            return 0.0
        price_str = price_str.lower().strip()
        if 'cr' in price_str:
            return float(price_str.replace('cr', '').strip())
        elif 'lakh' in price_str:
            return float(price_str.replace('lakh', '').strip()) / 100
        return 0.0
    
    def _add_history_features(self, df, batter_hist, bowler_hist, auction_year):
        """Add experience and historical WAR features."""
        print("  Adding historical features...")
        
        # Aggregate batter history
        batter_agg = batter_hist[batter_hist['season'] < auction_year].groupby('batter_name').agg({
            'season': 'count',  # seasons played
            'WAR': ['sum', 'mean', 'std'],
            'balls_faced': 'sum'
        }).reset_index()
        batter_agg.columns = ['player_name', 'bat_seasons', 'career_bat_war', 'avg_bat_war', 'bat_war_std', 'career_bat_balls']
        
        # Get last 2 seasons for recent form
        recent_batter = batter_hist[
            (batter_hist['season'] < auction_year) & 
            (batter_hist['season'] >= auction_year - 2)
        ].groupby('batter_name').agg({'WAR': 'mean'}).reset_index()
        recent_batter.columns = ['player_name', 'recent_bat_war']
        
        batter_agg = batter_agg.merge(recent_batter, on='player_name', how='left')
        
        # Aggregate bowler history
        bowler_agg = bowler_hist[bowler_hist['season'] < auction_year].groupby('bowler_name').agg({
            'season': 'count',
            'WAR': ['sum', 'mean', 'std'],
            'balls_bowled': 'sum'
        }).reset_index()
        bowler_agg.columns = ['player_name', 'bowl_seasons', 'career_bowl_war', 'avg_bowl_war', 'bowl_war_std', 'career_bowl_balls']
        
        recent_bowler = bowler_hist[
            (bowler_hist['season'] < auction_year) & 
            (bowler_hist['season'] >= auction_year - 2)
        ].groupby('bowler_name').agg({'WAR': 'mean'}).reset_index()
        recent_bowler.columns = ['player_name', 'recent_bowl_war']
        
        bowler_agg = bowler_agg.merge(recent_bowler, on='player_name', how='left')
        
        # Merge with main df
        df = df.merge(batter_agg, left_on='cricwar_name', right_on='player_name', 
                      how='left', suffixes=('', '_bat'))
        df = df.merge(bowler_agg, left_on='cricwar_name', right_on='player_name', 
                      how='left', suffixes=('', '_bowl'))
        
        # Combine batter + bowler features
        df['ipl_seasons'] = df[['bat_seasons', 'bowl_seasons']].max(axis=1).fillna(0)
        df['career_war'] = df[['career_bat_war', 'career_bowl_war']].sum(axis=1).fillna(0)
        df['avg_war'] = df[['avg_bat_war', 'avg_bowl_war']].mean(axis=1).fillna(0)
        df['war_consistency'] = df[['bat_war_std', 'bowl_war_std']].mean(axis=1).fillna(0)
        df['recent_war'] = df[['recent_bat_war', 'recent_bowl_war']].sum(axis=1).fillna(0)
        df['career_balls'] = df[['career_bat_balls', 'career_bowl_balls']].sum(axis=1).fillna(0)
        
        return df
    
    def _add_age_features(self, df, enriched, auction_year):
        """Add age from DOB."""
        print("  Adding age features...")
        
        # Parse DOB
        enriched_clean = enriched[enriched['dob'].notna()].copy()
        enriched_clean['dob'] = pd.to_datetime(enriched_clean['dob'], errors='coerce')
        
        # Calculate age at auction (November of auction year)
        auction_date = datetime(auction_year, 11, 1)
        enriched_clean['age'] = (auction_date - enriched_clean['dob']).dt.days / 365.25
        
        # Merge by name matching
        name_to_age = dict(zip(
            enriched_clean['unique_name'].str.lower().str.strip(),
            enriched_clean['age']
        ))
        
        # Also try full_name
        for _, row in enriched_clean.iterrows():
            if pd.notna(row['full_name']):
                name_to_age[str(row['full_name']).lower().strip()] = row['age']
        
        # Map ages
        def get_age(name):
            if pd.isna(name):
                return np.nan
            name_lower = str(name).lower().strip()
            return name_to_age.get(name_lower, np.nan)
        
        df['age'] = df['cricwar_name'].apply(get_age)
        
        # Fill missing ages with median
        median_age = df['age'].median()
        if pd.isna(median_age):
            median_age = 28  # Default median age
        df['age'] = df['age'].fillna(median_age)
        
        print(f"  Age range: {df['age'].min():.1f} - {df['age'].max():.1f}")
        return df
    
    def prepare_features(self, df, training=False):
        """Prepare feature matrix for training/prediction."""
        # Numeric features
        numeric_features = [
            'total_WAR',      # Current season WAR
            'age',            # Age at auction
            'ipl_seasons',    # Experience
            'career_war',     # Career cumulative
            'recent_war',     # Last 2 seasons
            'war_consistency', # Std dev (volatility)
            'is_overseas',    # Binary
            'base_price_lakh', # Base price (strong anchor)
        ]
        
        # Ensure columns exist
        for col in numeric_features:
            if col not in df.columns:
                df[col] = 0
        
        X = df[numeric_features].copy()
        
        # Consistent role categories
        ROLE_CATEGORIES = ['Allrounder', 'Batter', 'Bowler', 'Spinner', 'Wicketkeeper', 'Unknown']
        
        # Map role_category to standard set
        if 'role_category' in df.columns:
            df_role = df['role_category'].apply(self._standardize_role)
            role_dummies = pd.get_dummies(df_role, prefix='role')
            
            # Ensure all role columns exist
            for role in ROLE_CATEGORIES:
                col = f'role_{role}'
                if col not in role_dummies.columns:
                    role_dummies[col] = 0
            
            # Keep only standard columns in consistent order
            role_cols = [f'role_{r}' for r in ROLE_CATEGORIES]
            role_dummies = role_dummies[role_cols]
            
            X = pd.concat([X, role_dummies], axis=1)
        
        # Fill NaN
        X = X.fillna(0)
        
        if training:
            self.feature_columns = X.columns.tolist()
        else:
            # Reorder columns to match training
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]
        
        return X
    
    def _standardize_role(self, role):
        """Map diverse role strings to standard categories."""
        if pd.isna(role):
            return 'Batter'  # Safe default instead of Unknown
            
        r = str(role).lower().strip()
        
        # Standard categories
        if 'wicket' in r or 'keeper' in r or 'wk' in r:
            return 'Wicketkeeper'
        elif 'all' in r and 'round' in r:
            return 'Allrounder'
        elif 'bat' in r:
            return 'Batter'
        elif 'bowl' in r:
            return 'Bowler'
        elif 'spin' in r:
            return 'Bowler' # Spinner -> Bowler
            
        return 'Batter'  # Default to Batter for unknown/missing
    
    def train(self, df=None):
        """Train XGBoost model with cross-validation."""
        if df is None:
            df = self.load_training_data()
        
        print("\n" + "="*60)
        print("TRAINING XGBOOST PRICE MODEL")
        print("="*60)
        
        X = self.prepare_features(df, training=True)
        y = df['price_cr']
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_columns)}")
        
        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'reg_alpha': [0.1, 1.0],
            'reg_lambda': [1.0, 2.0],
        }
        
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        print("\nPerforming GridSearchCV...")
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=5, scoring='r2', 
            verbose=0, n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Final fit on all data
        self.model.fit(X, y)
        train_r2 = self.model.score(X, y)
        print(f"Training R²: {train_r2:.4f}")
        
        self.is_trained = True
        
        # Save feature importance
        self._save_feature_importance()
        
        return cv_scores.mean()
    
    def _save_feature_importance(self):
        """Plot and save feature importance."""
        output_dir = self.project_root / 'results' / 'analysis' / 'price_model'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save CSV
        importance.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'][:15], importance['importance'][:15])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Price Model - Top 15 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150)
        plt.close()
        
        print(f"\n✓ Feature importance saved to {output_dir}")
        
        # Print top features
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))
    
    def predict(self, df):
        """Predict prices for players in dataframe."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X = self.prepare_features(df)
        
        # Ensure same columns as training
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]
        
        predictions = self.model.predict(X)
        
        # Ensure minimum price of 0.2 Cr (20 Lakhs)
        predictions = np.maximum(predictions, 0.2)
        
        return predictions
    
    def compare_with_baseline(self, df):
        """Compare XGBoost with VOMAM/VOPE baselines."""
        from sklearn.linear_model import LinearRegression
        from scipy.optimize import curve_fit
        
        print("\n" + "="*60)
        print("COMPARING WITH BASELINE MODELS")
        print("="*60)
        
        # Prepare data
        X_xgb = self.prepare_features(df, training=True)
        # Handle NaN values in target which might cause training issues
        df['price_cr'] = df['price_cr'].fillna(0.0)
        y = df['price_cr']
        
        # --- VOMAM (Linear) ---
        X_vomam = df[['total_WAR', 'is_overseas']].copy()
        role_dummies = pd.get_dummies(df['role_category'], prefix='role')
        X_vomam = pd.concat([X_vomam, role_dummies], axis=1).fillna(0)
        
        vomam_model = LinearRegression()
        vomam_scores = cross_val_score(vomam_model, X_vomam, y, cv=5, scoring='r2')
        
        # --- VOPE (Power Law) ---
        def power_law(x, a, b):
            return a * np.power(np.maximum(x, 0.01), b)
        
        # Simple R² for VOPE (can't use CV easily)
        vope_data = df[df['total_WAR'] > 0]
        try:
            popt, _ = curve_fit(power_law, vope_data['total_WAR'], vope_data['price_cr'], 
                               p0=[1, 1], maxfev=5000)
            vope_pred = power_law(vope_data['total_WAR'], *popt)
            ss_res = np.sum((vope_data['price_cr'] - vope_pred) ** 2)
            ss_tot = np.sum((vope_data['price_cr'] - np.mean(vope_data['price_cr'])) ** 2)
            vope_r2 = 1 - (ss_res / ss_tot)
        except:
            vope_r2 = 0.0
        
        # --- XGBoost ---
        xgb_scores = cross_val_score(self.model, X_xgb, y, cv=5, scoring='r2')
        
        # Results
        results = pd.DataFrame({
            'Model': ['XGBoost (Ours)', 'VOMAM (Linear)', 'VOPE (Power Law)'],
            'CV R²': [xgb_scores.mean(), vomam_scores.mean(), vope_r2],
            'CV Std': [xgb_scores.std()*2, vomam_scores.std()*2, 0],
            'Improvement vs VOMAM': [
                f"+{(xgb_scores.mean() - vomam_scores.mean())*100:.1f}%",
                "Baseline",
                f"{(vope_r2 - vomam_scores.mean())*100:+.1f}%"
            ]
        })
        
        print("\nModel Comparison:")
        print(results.to_string(index=False))
        
        # Save comparison
        output_dir = self.project_root / 'results' / 'analysis' / 'price_model'
        results.to_csv(output_dir / 'model_comparison.csv', index=False)
        
        return results
    
    def predict_2026_auction(self):
        """Score 2026 auction pool with XGBoost predictions."""
        print("\n" + "="*60)
        print("PREDICTING 2026 AUCTION PRICES")
        print("="*60)
        
        # Load auction pool
        auction_df = pd.read_csv(self.project_root / 'data' / 'ipl_2026_auction_enriched.csv')
        print(f"Loaded {len(auction_df)} auction players")
        
        # Load V9 projections
        v9_bat = pd.read_csv(self.project_root / 'results' / 'WARprojections' / 'v9_production' / 'batter_projections_2026_v9prod.csv')
        v9_bowl = pd.read_csv(self.project_root / 'results' / 'WARprojections' / 'v9_production' / 'bowler_projections_2026_v9prod.csv')
        
        # Load WAR history for experience features
        batter_hist = pd.read_csv(self.project_root / 'data' / 'batter_war_full_history.csv')
        bowler_hist = pd.read_csv(self.project_root / 'data' / 'bowler_war_full_history.csv')
        
        # Load enriched for DOB
        try:
            enriched = pd.read_csv(self.project_root / 'data' / 'all_players_enriched.csv')
        except:
            enriched = pd.DataFrame()
        
        # Build feature dataframe for auction players
        results = []
        for _, row in auction_df.iterrows():
            player_name = row.get('PLAYER') or row.get('name')
            cricsheet_id = row.get('cricsheet_id')
            
            # Get WAR projection
            war_proj = 0
            war_source = 'Unknown'
            
            # Try V9 production first
            v9_match = v9_bat[v9_bat['batter_id'] == cricsheet_id]
            if len(v9_match) > 0:
                war_proj = v9_match['projected_war_2026'].values[0]
                war_source = 'V9_Bat'
            else:
                v9_match = v9_bowl[v9_bowl['bowler_id'] == cricsheet_id]
                if len(v9_match) > 0:
                    war_proj = v9_match['projected_war_2026'].values[0]
                    war_source = 'V9_Bowl'
            
            # Get experience
            bat_exp = batter_hist[batter_hist['batter_id'] == cricsheet_id]
            bowl_exp = bowler_hist[bowler_hist['bowler_id'] == cricsheet_id]
            
            ipl_seasons = max(
                bat_exp['season'].nunique() if len(bat_exp) > 0 else 0,
                bowl_exp['season'].nunique() if len(bowl_exp) > 0 else 0
            )
            career_war = (
                (bat_exp['WAR'].sum() if len(bat_exp) > 0 else 0) +
                (bowl_exp['WAR'].sum() if len(bowl_exp) > 0 else 0)
            )
            
            # Recent WAR (last 2 seasons)
            recent_bat = bat_exp[bat_exp['season'] >= 2024]['WAR'].mean() if len(bat_exp[bat_exp['season'] >= 2024]) > 0 else 0
            recent_bowl = bowl_exp[bowl_exp['season'] >= 2024]['WAR'].mean() if len(bowl_exp[bowl_exp['season'] >= 2024]) > 0 else 0
            recent_war = recent_bat + recent_bowl if pd.notna(recent_bat + recent_bowl) else 0
            
            # Age
            age = 28  # Default
            if not enriched.empty and pd.notna(row.get('dob')):
                try:
                    dob = pd.to_datetime(row['dob'])
                    age = (datetime(2025, 11, 1) - dob).days / 365.25
                except:
                    pass
            
            # Overseas
            country = row.get('COUNTRY') or row.get('country', '')
            is_overseas = 0 if str(country).lower() in ['india', 'ind'] else 1
            
            # Role
            role = row.get('playing_role', 'Unknown')
            
            results.append({
                'player': player_name,
                'country': country,
                'role': role,
                'base_price_lakh': row.get('BASE PRICE (INR LAKH)', 0),
                'total_WAR': war_proj,
                'war_source': war_source,
                'age': age,
                'ipl_seasons': ipl_seasons,
                'career_war': career_war,
                'recent_war': recent_war,
                'war_consistency': 0,  # Not calculated for simplicity
                'is_overseas': is_overseas,
                'role_category': self._map_role(role),
            })
        
        pred_df = pd.DataFrame(results)
        
        # Predict prices
        pred_df['predicted_price_cr'] = self.predict(pred_df)
        
        # Sort by predicted price
        pred_df = pred_df.sort_values('predicted_price_cr', ascending=False)
        
        # Save
        output_dir = self.project_root / 'results' / 'analysis' / 'price_model'
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(output_dir / 'auction_prices_2026_xgb.csv', index=False)
        
        print(f"\n✓ Saved predictions to {output_dir / 'auction_prices_2026_xgb.csv'}")
        
        # Show top predictions
        print("\nTop 20 Predicted Prices:")
        print(pred_df[['player', 'role', 'total_WAR', 'age', 'ipl_seasons', 'predicted_price_cr']].head(20).to_string(index=False))
        
        return pred_df
    
    def _map_role(self, role):
        """Map playing_role to role_category (use same logic as _standardize_role)."""
        return self._standardize_role(role)


def main():
    """Main entry point."""
    print("="*60)
    print("XGBOOST IPL AUCTION PRICE PREDICTION MODEL")
    print("="*60)
    
    model = XGBoostPriceModel()
    
    # 1. Load and train
    training_df = model.load_training_data(auction_year=2025)
    model.train(training_df)
    
    # 2. Compare with baselines
    model.compare_with_baseline(training_df)
    
    # 3. Predict 2026 auction
    predictions = model.predict_2026_auction()
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()

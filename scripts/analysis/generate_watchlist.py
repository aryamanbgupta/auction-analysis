import pandas as pd
from pathlib import Path
import sys

# Add scripts/analysis to path to import ValuationEngine
sys.path.append(str(Path(__file__).parent))
from valuation_engine import ValuationEngine

def main():
    root = Path(__file__).parent.parent.parent
    output_dir = root / 'results' / 'analysis' / 'strategic' / 'valued'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Engine
    engine = ValuationEngine(root)
    engine.load_data()
    engine.train_models()
    
    # Load Stats
    batters_path = root / 'results' / 'analysis' / 'auction_pool' / 'auction_pool_batters.csv'
    bowlers_path = root / 'results' / 'analysis' / 'auction_pool' / 'auction_pool_bowlers.csv'
    
    df_batters = pd.read_csv(batters_path).set_index('batter_name')
    df_bowlers = pd.read_csv(bowlers_path).set_index('bowler_name')
    
    # Target List (User Name -> Data Name)
    targets = {
        'Tim Seifert': 'TL Seifert',
        'Spencer Johnson': 'SH Johnson',
        'Jason Behrendorff': 'JP Behrendorff',
        'Naveen-ul-Haq': 'Naveen-ul-Haq',
        'Chetan Sakariya': 'C Sakariya',
        'Akash Madhwal': 'Akash Madhwal',
        'Atharva Taide': 'Atharva Taide',
        'Abhinav Manohar': 'A Manohar'
    }
    
    results = []
    
    for user_name, data_name in targets.items():
        row = {'User_Name': user_name, 'Player': data_name}
        
        # 1. Get Valuation (WAR + Price)
        # Need role/country for price prediction
        # Try to find in metadata
        meta = engine.metadata[engine.metadata['name_norm'] == str(data_name).strip().lower()]
        if not meta.empty:
            role = meta.iloc[0]['role_category']
            country = meta.iloc[0]['country']
            is_overseas = 0 if str(country).lower() in ['india', 'ind'] else 1
        else:
            # Fallback defaults
            role = 'Unknown'
            country = 'Unknown'
            is_overseas = 1 # Assume overseas if unknown for safety
            
        # Get WAR
        war = engine.projections[engine.projections['name_norm'] == str(data_name).strip().lower()]['war_2026'].values
        war_val = war[0] if len(war) > 0 else 0.0
        
        vomam, vope = engine.predict_price(war_val, is_overseas, role)
        
        row['WAR_2026'] = round(war_val, 2)
        row['Est_Price_VOMAM'] = round(vomam, 2)
        row['Est_Price_VOPE'] = round(vope, 2)
        row['Est_Price_Avg'] = round((vomam + vope) / 2, 2)
        
        # 2. Get Stats
        # Check Batters
        if data_name in df_batters.index:
            b_stats = df_batters.loc[data_name]
            row['Type'] = 'Batter'
            row['PP_SR'] = round(b_stats.get('SR_powerplay', 0), 2)
            row['PP_RAA'] = round(b_stats.get('RAA_powerplay', 0), 2)
            row['Middle_RAA'] = round(b_stats.get('RAA_middle', 0), 2)
        
        # Check Bowlers (Overwrite Type if primarily a bowler or found there)
        if data_name in df_bowlers.index:
            bo_stats = df_bowlers.loc[data_name]
            # If already found in batters, check which role is dominant? 
            # For now, if in bowlers, add bowler stats.
            if 'Type' not in row: row['Type'] = 'Bowler'
            
            row['PP_Wickets'] = int(bo_stats.get('Wickets_powerplay', 0))
            row['PP_RAA_Bowl'] = round(bo_stats.get('RAA_powerplay', 0), 2)
            
            # Death Stats
            row['Death_Wickets'] = int(bo_stats.get('Wickets_death', 0))
            row['Death_Econ'] = round(bo_stats.get('Econ_death', 0), 2)
            row['Death_RAA'] = round(bo_stats.get('RAA_death', 0), 2)


        results.append(row)
        
    # Create DataFrame
    df_res = pd.DataFrame(results)
    
    # Save
    save_path = output_dir / 'valued_watchlist.csv'
    df_res.to_csv(save_path, index=False)
    print(f"Saved watchlist to {save_path}")
    print(df_res)

if __name__ == "__main__":
    main()


import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    
    df = pd.read_csv(metadata_file)
    
    # Manual corrections dictionary
    corrections = {
        'P Simran Singh': 'India',
        'AM Rahane': 'India',
        'S Dube': 'India',
        'RG Sharma': 'India',
        'Ishan Kishan': 'India',
        'SN Thakur': 'India',
        'J Suchith': 'India',
        'DL Chahar': 'India',
        'Arshad Khan': 'India',
        'T Stubbs': 'South Africa',
        'R Ravindra': 'New Zealand',
        'Rasikh Salam': 'India',
        'MS Bhandage': 'India',
        'DJ Willey': 'England',
        'Mohammed Shami': 'India',
        'Yudhvir Singh': 'India',
        'Ravi Bishnoi': 'India',
        'Atharva Taide': 'India',
        'PVSN Raju': 'India',
        'S Gopal': 'India',
        'Urvil Patel': 'India',
        'SH Johnson': 'Australia',
        'Sediqullah Atal': 'Afghanistan',
        'Arshad Khan (2)': 'India',
        'LB Williams': 'South Africa',
        'Swapnil Singh': 'India',
        'MJ Suthar': 'India',
        'PJ Sangwan': 'India',
        'JA Richardson': 'Australia',
        'JDS Neesham': 'New Zealand',
        'NM Coulter-Nile': 'Australia',
        'KR Sen': 'India',
        'KM Jadhav': 'India',
        'Harsh Dubey': 'India',
        'W O\'Rourke': 'New Zealand'
    }
    
    # Apply corrections
    for name, country in corrections.items():
        mask = df['player_name'] == name
        if mask.any():
            df.loc[mask, 'country'] = country
            print(f"Updated {name} -> {country}")
        else:
            print(f"Warning: {name} not found in metadata")
            
    # Save back
    df.to_csv(metadata_file, index=False)
    print(f"\nSaved updated metadata to {metadata_file}")
    
    # Verify
    print("\nRemaining Unknowns:")
    print(df[df['country'].isna()]['player_name'].tolist())

if __name__ == "__main__":
    main()

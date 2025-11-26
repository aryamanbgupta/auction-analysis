"""
Fetch player metadata using existing player info CSV.

This script:
1. Loads the list of unique players from IPL data
2. Uses existing players_info.csv (from your cricinfo scraper)
3. Maps player IDs to metadata (batting style, bowling style, playing role)
4. Saves to data/player_metadata.csv
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_existing_player_info():
    """
    Load player metadata from data/players_info.csv.

    Returns:
        pandas DataFrame with player metadata
    """
    # Path to players_info.csv
    project_root = Path(__file__).parent.parent
    player_info_file = project_root / 'data' / 'players_info.csv'

    if not player_info_file.exists():
        print(f"Warning: players_info.csv not found at {player_info_file}")
        print("Creating empty metadata template...")
        return pd.DataFrame(columns=['identifier', 'name', 'full_name', 'batting_styles', 'bowling_styles', 'playing_roles'])

    print(f"Loading player info from {player_info_file}...")
    player_info = pd.read_csv(player_info_file)
    
    # Clean column names
    player_info.columns = player_info.columns.str.strip()
    
    # Clean IDs: convert to string and remove .0
    if 'identifier' in player_info.columns:
        player_info['identifier'] = player_info['identifier'].astype(str).str.replace(r'\.0$', '', regex=True)
        
    # Replace "Not found" with None
    player_info = player_info.replace('Not found', None)

    print(f"✓ Loaded {len(player_info)} players from existing data")

    return player_info


def map_player_ids(ipl_df, player_meta_df):
    """
    Map player IDs from IPL data to player metadata using ID matching.

    Args:
        ipl_df: IPL ball-by-ball DataFrame
        player_meta_df: Player metadata DataFrame (from players_info.csv)

    Returns:
        DataFrame with mapped player information
    """
    # Get unique players from IPL data
    batters = ipl_df[['batter_id', 'batter_name']].drop_duplicates()
    batters.columns = ['player_id', 'player_name']

    bowlers = ipl_df[['bowler_id', 'bowler_name']].drop_duplicates()
    bowlers.columns = ['player_id', 'player_name']

    non_strikers = ipl_df[['non_striker_id', 'non_striker_name']].drop_duplicates()
    non_strikers.columns = ['player_id', 'player_name']

    # Combine and remove duplicates
    all_players = pd.concat([batters, bowlers, non_strikers]).drop_duplicates()
    all_players = all_players[all_players['player_id'].notna()]
    
    # Clean IPL IDs
    all_players['player_id'] = all_players['player_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    print(f"\n✓ Found {len(all_players)} unique players in IPL data")

    # If no metadata loaded, return just the player list
    if len(player_meta_df) == 0:
        print("⚠ No existing player metadata found - creating basic template")
        return all_players

    # Merge with metadata using 'identifier' from info file and 'player_id' from IPL
    # Rename info columns to match expected output format
    player_meta_df = player_meta_df.rename(columns={
        'identifier': 'player_id',
        'playing_roles': 'playing_role',
        'batting_styles': 'batting_style',
        'bowling_styles': 'bowling_style'
    })
    
    # Ensure player_id is string in metadata
    if 'player_id' in player_meta_df.columns:
        player_meta_df['player_id'] = player_meta_df['player_id'].astype(str)

    # Merge
    merged = all_players.merge(
        player_meta_df,
        on='player_id',
        how='left'
    )
    
    # Load and merge manual updates
    project_root = Path(__file__).parent.parent
    manual_update_file = project_root / 'data' / 'updated_players_export.csv'
    
    if manual_update_file.exists():
        print(f"\nLoading manual updates from {manual_update_file}...")
        manual_updates = pd.read_csv(manual_update_file)
        
        # Ensure player_id is string
        if 'player_id' in manual_updates.columns:
            manual_updates['player_id'] = manual_updates['player_id'].astype(str)
            
        # Merge manual updates
        merged = merged.merge(
            manual_updates[['player_id', 'new_role_category']],
            on='player_id',
            how='left'
        )
        
        # Update playing_role where available
        mask = merged['new_role_category'].notna()
        merged.loc[mask, 'playing_role'] = merged.loc[mask, 'new_role_category']
        
        # Drop temporary column
        merged = merged.drop(columns=['new_role_category'])
        
        print(f"✓ Applied manual updates for {mask.sum()} players")

    # Fill missing metadata
    if 'playing_role' in merged.columns:
        matched = merged['playing_role'].notna().sum()
        total = len(merged)
        print(f"✓ Matched roles for {matched}/{total} players ({matched/total*100:.1f}%)")

    # Merge country from player_metadata_full.csv if available
    project_root = Path(__file__).parent.parent
    full_meta_file = project_root / 'data' / 'player_metadata_full.csv'
    
    if full_meta_file.exists():
        print(f"\nLoading country data from {full_meta_file}...")
        full_meta = pd.read_csv(full_meta_file)
        
        # Ensure player_id is string
        if 'player_id' in full_meta.columns:
            full_meta['player_id'] = full_meta['player_id'].astype(str)
            
            # Merge country
            if 'country' in full_meta.columns:
                merged = merged.merge(
                    full_meta[['player_id', 'country']],
                    on='player_id',
                    how='left'
                )
                print(f"✓ Merged country data for {merged['country'].notna().sum()} players")
    
    return merged


def categorize_players(metadata_df):
    """
    Add categorized columns for analysis.

    Args:
        metadata_df: Player metadata DataFrame

    Returns:
        DataFrame with additional categorized columns
    """
    print("\nCategorizing players...")

    # Add missing columns with defaults if they don't exist
    if 'batting_style' not in metadata_df.columns:
        metadata_df['batting_style'] = None
    if 'bowling_style' not in metadata_df.columns:
        metadata_df['bowling_style'] = None
    if 'playing_role' not in metadata_df.columns:
        metadata_df['playing_role'] = None

    # Batting hand: Left (L) or Right (R)
    metadata_df['batting_hand'] = metadata_df['batting_style'].apply(
        lambda x: 'LHB' if pd.notna(x) and 'L' in str(x).upper() else ('RHB' if pd.notna(x) else 'unknown')
    )

    # Bowling type: pace or spin
    def categorize_bowling(style):
        if pd.isna(style):
            return 'pace' # Default to pace as requested
        style_str = str(style).upper()

        # Spin indicators
        if any(indicator in style_str for indicator in ['SL', 'OB', 'LB', 'LBG', 'SLOW', 'LEG', 'OFF', 'BREAK', 'GOOGLY', 'ORTHODOX', 'CHINAMAN']):
            return 'spin'

        # Pace indicators
        if any(indicator in style_str for indicator in ['FAST', 'MEDIUM', 'F', 'M', 'SEAM']):
            return 'pace'

        return 'pace' # Default to pace for unknown styles

    metadata_df['bowling_type'] = metadata_df['bowling_style'].apply(categorize_bowling)

    # Bowling arm: Left or Right
    metadata_df['bowling_arm'] = metadata_df['bowling_style'].apply(
        lambda x: 'left-arm' if pd.notna(x) and 'L' in str(x).upper() else ('right-arm' if pd.notna(x) else 'unknown')
    )

    # Standardize playing role
    def standardize_role(row):
        role = row['playing_role']
        b_type = row['bowling_type']
        
        if pd.isna(role):
            return 'Middle-order Batter' # Default fallback
            
        role_str = str(role).lower()

        # Wicketkeeper
        if 'keep' in role_str or 'wicket' in role_str:
            return 'Wicketkeeper'
            
        # Allrounder
        if 'all' in role_str or 'rounder' in role_str:
            return 'Allrounder'
            
        # Batters
        if 'bat' in role_str:
            if 'open' in role_str or 'top' in role_str:
                return 'Top-order Batter'
            elif 'middle' in role_str:
                return 'Middle-order Batter'
            else:
                # Generic 'Batter' -> Middle-order (smaller category)
                return 'Middle-order Batter'
                
        # Bowlers
        if 'bowl' in role_str:
            if b_type == 'spin':
                return 'Spinner'
            else:
                return 'Pacer'
                
        return 'Middle-order Batter' # Fallback

    metadata_df['role_category'] = metadata_df.apply(standardize_role, axis=1)

    # Statistics
    print("\nPlayer categories:")
    print("\nBatting hand:")
    print(metadata_df['batting_hand'].value_counts())

    print("\nBowling type:")
    print(metadata_df['bowling_type'].value_counts())

    print("\nPlaying role:")
    print(metadata_df['role_category'].value_counts())

    return metadata_df


def main():
    """Fetch and process player metadata."""

    # Paths
    project_root = Path(__file__).parent.parent
    ipl_file = project_root / 'data' / 'ipl_matches.parquet'
    output_file = project_root / 'data' / 'player_metadata.csv'

    print("="*60)
    print("PROCESSING PLAYER METADATA")
    print("="*60)

    # Load IPL data
    print(f"\nLoading IPL data from {ipl_file}...")
    ipl_df = pd.read_parquet(ipl_file)
    print(f"✓ Loaded {len(ipl_df):,} balls")

    # Load existing player info
    player_meta_df = load_existing_player_info()

    # Map player IDs
    metadata = map_player_ids(ipl_df, player_meta_df)

    # Categorize players
    metadata = categorize_players(metadata)

    # Save to CSV
    print(f"\nSaving to {output_file}...")
    metadata.to_csv(output_file, index=False)
    print(f"✓ Saved {len(metadata)} players")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total players: {len(metadata)}")

    # Check if we have the 'name' column or use a different one
    if 'name' in metadata.columns:
        print(f"With metadata: {metadata['name'].notna().sum()}")
        print(f"Missing metadata: {metadata['name'].isna().sum()}")

        # Show sample of players with metadata
        if metadata['name'].notna().any():
            print("\nSample of players with metadata:")
            sample = metadata[metadata['name'].notna()].head(10)
            cols = [c for c in ['player_name', 'country', 'role_category', 'batting_hand', 'bowling_type'] if c in sample.columns]
            print(sample[cols].to_string(index=False))

        # Show sample of players missing metadata
        if metadata['name'].isna().any():
            print("\nSample of players MISSING metadata (need manual lookup):")
            missing = metadata[metadata['name'].isna()].head(10)
            print(missing[['player_id', 'player_name']].to_string(index=False))
    else:
        # Use full_name if available
        if 'full_name' in metadata.columns:
            print(f"With metadata: {metadata['full_name'].notna().sum()}")
            print(f"Missing metadata: {metadata['full_name'].isna().sum()}")

    print("\n" + "="*60)
    print("✓ PLAYER METADATA PROCESSING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

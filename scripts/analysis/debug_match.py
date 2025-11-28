import pandas as pd

def normalize_name(name):
    if pd.isna(name):
        return ""
    return str(name).strip().lower()

def is_match(dataset_name, retention_name):
    d_norm = normalize_name(dataset_name)
    r_norm = normalize_name(retention_name)
    
    print(f"Comparing '{d_norm}' vs '{r_norm}'")
    
    if d_norm == r_norm:
        return True
        
    d_parts = d_norm.split()
    r_parts = r_norm.split()
    
    if not d_parts or not r_parts:
        return False
        
    # Last name must match
    if d_parts[-1] != r_parts[-1]:
        print(f"  Last name mismatch: {d_parts[-1]} != {r_parts[-1]}")
        return False
        
    # First initial must match
    if d_parts[0][0] != r_parts[0][0]:
        print(f"  Initial mismatch: {d_parts[0][0]} != {r_parts[0][0]}")
        return False
        
    print("  MATCH!")
    return True

# Test cases
is_match("SP Narine", "Sunil Narine")
is_match("JJ Bumrah", "Jasprit Bumrah")
is_match("SA Yadav", "Suryakumar Yadav")


import pandas as pd
from pathlib import Path

def main():
    project_root = Path('.').resolve()
    metadata_file = project_root / 'data' / 'player_metadata.csv'
    
    df = pd.read_csv(metadata_file)
    
    # Manual corrections dictionary
    # Format: 'Player Name': {'role_category': '...', 'bowling_type': '...'}
    corrections = {
        'Kuldeep Yadav': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Noor Ahmad': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Rashid Khan': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'AR Patel': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'Sandeep Sharma': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Dhruv Jurel': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'}, # WK usually listed as pace/unknown
        'AD Russell': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'Suyash Sharma': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'M Shahrukh Khan': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'R Sai Kishore': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'CV Varun': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'SP Narine': {'role_category': 'Allrounder', 'bowling_type': 'spin'}, # Or Spinner? He opens batting. Allrounder fits better given usage.
        'RA Jadeja': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'HH Pandya': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'MP Stoinis': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'LS Livingstone': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'GJ Maxwell': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'MM Ali': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'Washington Sundar': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'R Ashwin': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'YS Chahal': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Ravi Bishnoi': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'PP Chawla': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Avesh Khan': {'role_category': 'Pacer', 'bowling_type': 'pace'}, # Was Batting Allrounder?
        'Arshdeep Singh': {'role_category': 'Pacer', 'bowling_type': 'pace'}, # Was Middle-order Batter?
        'Abhishek Sharma': {'role_category': 'Allrounder', 'bowling_type': 'spin'}, # Bowls spin, bats top order. Or Top-order Batter? Let's say Allrounder or Top-order. He's mainly a batter who bowls.
        'YBK Jaiswal': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'},
        'Shubman Gill': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'}, # Was Allrounder?
        'Tilak Varma': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'}, # Was Allrounder?
        'R Shepherd': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'M Jansen': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'PJ Cummins': {'role_category': 'Allrounder', 'bowling_type': 'pace'}, # Or Pacer? He bats well.
        'Mohammed Siraj': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'B Kumar': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Mukesh Kumar': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Yash Dayal': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Harshit Rana': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Vijaykumar Vyshak': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'TU Deshpande': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'KK Nair': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'N Rana': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'A Badoni': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'Sameer Rizvi': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'Abdul Samad': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'Aniket Verma': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'Ashutosh Sharma': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'Abishek Porel': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Ishan Kishan': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'}, # Was Middle-order Batter? He opens.
        'KL Rahul': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'}, # Was Middle-order?
        'JC Buttler': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'SV Samson': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'RR Pant': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Q de Kock': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'JM Bairstow': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'DP Conway': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'MS Dhoni': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Dinesh Karthik': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'WP Saha': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'KS Bharat': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Anuj Rawat': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Jitesh Sharma': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Prabhsimran Singh': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'}, # P Simran Singh
        'P Simran Singh': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Rahmanullah Gurbaz': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Phil Salt': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'PD Salt': {'role_category': 'Wicketkeeper', 'bowling_type': 'pace'},
        'Shreyas Iyer': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'}, # SS Iyer
        'SS Iyer': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'},
        'Suryakumar Yadav': {'role_category': 'Top-order Batter', 'bowling_type': 'pace'}, # SA Yadav
        'SA Yadav': {'role_category': 'Top-order Batter', 'bowling_type': 'pace'},
        'Rohit Sharma': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'}, # RG Sharma
        'RG Sharma': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'},
        'Virat Kohli': {'role_category': 'Top-order Batter', 'bowling_type': 'pace'}, # V Kohli
        'V Kohli': {'role_category': 'Top-order Batter', 'bowling_type': 'pace'},
        'Faf du Plessis': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'},
        'F du Plessis': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'},
        'David Warner': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'},
        'DA Warner': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'},
        'Ruturaj Gaikwad': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'}, # RD Gaikwad
        'RD Gaikwad': {'role_category': 'Top-order Batter', 'bowling_type': 'spin'},
        'Rinku Singh': {'role_category': 'Middle-order Batter', 'bowling_type': 'spin'},
        'Shivam Dube': {'role_category': 'Allrounder', 'bowling_type': 'pace'}, # S Dube
        'S Dube': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'Hardik Pandya': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'Krunal Pandya': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'KH Pandya': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'Deepak Chahar': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'DL Chahar': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Shardul Thakur': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'SN Thakur': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'T Natarajan': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Mohit Sharma': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'MM Sharma': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Trent Boult': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'TA Boult': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Kagiso Rabada': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'K Rabada': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Anrich Nortje': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'A Nortje': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Matheesha Pathirana': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'M Pathirana': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Mustafizur Rahman': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Mitchell Starc': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'MA Starc': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Pat Cummins': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'Sam Curran': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'SM Curran': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'Cameron Green': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'C Green': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'Daryll Mitchell': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'DJ Mitchell': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'Rachin Ravindra': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'R Ravindra': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'Azmatullah Omarzai': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'A Omarzai': {'role_category': 'Allrounder', 'bowling_type': 'pace'},
        'Wanindu Hasaranga': {'role_category': 'Allrounder', 'bowling_type': 'spin'}, # Or Spinner?
        'PWH de Silva': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
        'Maheesh Theekshana': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'M Theekshana': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Mujeeb Ur Rahman': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Adam Zampa': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'A Zampa': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Tabraiz Shamsi': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Ishant Sharma': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'I Sharma': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Umesh Yadav': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'UT Yadav': {'role_category': 'Pacer', 'bowling_type': 'pace'},
        'Varun Chakravarthy': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Rahul Chahar': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'RD Chahar': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Mayank Markande': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'M Markande': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Shreyas Gopal': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'S Gopal': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Karn Sharma': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'KV Sharma': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Amit Mishra': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'A Mishra': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Piyush Chawla': {'role_category': 'Spinner', 'bowling_type': 'spin'},
        'Ravichandran Ashwin': {'role_category': 'Allrounder', 'bowling_type': 'spin'},
    }
    
    count = 0
    for name, data in corrections.items():
        mask = df['player_name'] == name
        if mask.any():
            df.loc[mask, 'role_category'] = data['role_category']
            df.loc[mask, 'bowling_type'] = data['bowling_type']
            # Also update playing_role to match category for consistency
            df.loc[mask, 'playing_role'] = data['role_category']
            count += 1
            # print(f"Updated {name}")
            
    print(f"Updated roles for {count} players")
    
    # Save back
    df.to_csv(metadata_file, index=False)
    print(f"Saved updated metadata to {metadata_file}")

if __name__ == "__main__":
    main()

"""
Compare our squad CSV against IPL official website data and fix discrepancies.
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / 'data'

# Official IPL website squads (scraped from iplt20.com)
OFFICIAL = {
    'CSK': [
        'Ruturaj Gaikwad', 'MS Dhoni', 'Sanju Samson', 'Dewald Brevis',
        'Ayush Mhatre', 'Kartik Sharma', 'Sarfaraz Khan', 'Urvil Patel',
        'Anshul Kamboj', 'Jamie Overton', 'Ramakrishna Ghosh', 'Prashant Veer',
        'Matthew Short', 'Aman Khan', 'Zak Foulkes', 'Shivam Dube',
        'Khaleel Ahmed', 'Noor Ahmad', 'Mukesh Choudhary', 'Nathan Ellis',
        'Shreyas Gopal', 'Gurjapneet Singh', 'Akeal Hosein', 'Matt Henry',
        'Rahul Chahar',
    ],
    'DC': [
        'KL Rahul', 'Karun Nair', 'David Miller', 'Ben Duckett',
        'Pathum Nissanka', 'Sahil Parakh', 'Prithvi Shaw', 'Abishek Porel',
        'Tristan Stubbs', 'Axar Patel', 'Sameer Rizvi', 'Ashutosh Sharma',
        'Vipraj Nigam', 'Ajay Mandal', 'Tripurana Vijay', 'Madhav Tiwari',
        'Auqib Dar', 'Nitish Rana', 'Mitchell Starc', 'T. Natarajan',
        'Mukesh Kumar', 'Dushmantha Chameera', 'Lungisani Ngidi',
        'Kyle Jamieson', 'Kuldeep Yadav',
    ],
    'MI': [
        'Rohit Sharma', 'Surya Kumar Yadav', 'Robin Minz', 'Sherfane Rutherford',
        'Ryan Rickelton', 'Quinton de Kock', 'Danish Malewar', 'N. Tilak Varma',
        'Hardik Pandya', 'Naman Dhir', 'Mitchell Santner', 'Raj Angad Bawa',
        'Atharva Ankolekar', 'Mayank Rawat', 'Corbin Bosch', 'Will Jacks',
        'Shardul Thakur', 'Trent Boult', 'Mayank Markande', 'Deepak Chahar',
        'Ashwani Kumar', 'Raghu Sharma', 'Mohammad Izhar', 'Allah Ghazanfar',
        'Jasprit Bumrah',
    ],
    'RCB': [
        'Rajat Patidar', 'Devdutt Padikkal', 'Virat Kohli', 'Phil Salt',
        'Jitesh Sharma', 'Jordan Cox', 'Krunal Pandya', 'Swapnil Singh',
        'Tim David', 'Romario Shepherd', 'Jacob Bethell', 'Venkatesh Iyer',
        'Satvik Deswal', 'Mangesh Yadav', 'Vicky Ostwal', 'Vihaan Malhotra',
        'Kanishk Chouhan', 'Josh Hazlewood', 'Rasikh Dar', 'Suyash Sharma',
        'Bhuvneshwar Kumar', 'Nuwan Thushara', 'Abhinandan Singh',
        'Jacob Duffy', 'Yash Dayal',
    ],
    'KKR': [
        'Ajinkya Rahane', 'Rinku Singh', 'Angkrish Raghuvanshi', 'Manish Pandey',
        'Cameron Green', 'Finn Allen', 'Tejasvi Singh', 'Rahul Tripathi',
        'Tim Seifert', 'Rovman Powell', 'Anukul Roy', 'Sarthak Ranjan',
        'Daksh Kamra', 'Rachin Ravindra', 'Ramandeep Singh', 'Blessing Muzarabani',
        'Vaibhav Arora', 'Matheesha Pathirana', 'Kartik Tyagi', 'Prashant Solanki',
        'Akash Deep', 'Harshit Rana', 'Umran Malik', 'Sunil Narine',
        'Varun Chakaravarthy',
    ],
    'SRH': [
        'Ishan Kishan', 'Aniket Verma', 'Smaran Ravichandran', 'Salil Arora',
        'Heinrich Klaasen', 'Travis Head', 'Harshal Patel', 'Kamindu Mendis',
        'Harsh Dubey', 'Brydon Carse', 'Shivang Kumar', 'Krains Fuletra',
        'Liam Livingstone', 'Jack Edwards', 'Abhishek Sharma',
        'Nitish Kumar Reddy', 'Pat Cummins', 'Zeeshan Ansari',
        'Jaydev Unadkat', 'Eshan Malinga', 'Sakib Hussain', 'Onkar Tarmale',
        'Amit Kumar', 'Praful Hinge', 'Shivam Mavi',
    ],
    'RR': [
        'Riyan Parag', 'Shubham Dubey', 'Vaibhav Suryavanshi', 'Donovan Ferreira',
        'Lhuan-dre Pretorius', 'Ravi Singh', 'Aman Rao Perala',
        'Shimron Hetmyer', 'Yashasvi Jaiswal', 'Dhruv Jurel',
        'Yudhvir Singh Charak', 'Ravindra Jadeja', 'Sam Curran',
        'Jofra Archer', 'Tushar Deshpande', 'Kwena Maphaka', 'Ravi Bishnoi',
        'Sushant Mishra', 'Yash Raj Punja', 'Vignesh Puthur',
        'Brijesh Sharma', 'Adam Milne', 'Kuldeep Sen', 'Sandeep Sharma',
        'Nandre Burger',
    ],
    'PBKS': [
        'Shreyas Iyer', 'Nehal Wadhera', 'Vishnu Vinod', 'Harnoor Pannu',
        'Pyla Avinash', 'Prabhsimran Singh', 'Shashank Singh',
        'Marcus Stoinis', 'Harpreet Brar', 'Marco Jansen',
        'Azmatullah Omarzai', 'Priyansh Arya', 'Musheer Khan',
        'Suryansh Shedge', 'Mitch Owen', 'Cooper Connolly', 'Ben Dwarshuis',
        'Arshdeep Singh', 'Yuzvendra Chahal', 'Vyshak Vijaykumar',
        'Yash Thakur', 'Xavier Bartlett', 'Pravin Dubey', 'Vishal Nishad',
        'Lockie Ferguson',
    ],
    'GT': [
        'Shubman Gill', 'Jos Buttler', 'Kumar Kushagra', 'Anuj Rawat',
        'Tom Banton', 'Glenn Phillips', 'Nishant Sindhu', 'Washington Sundar',
        'Mohd. Arshad Khan', 'Sai Kishore', 'Jayant Yadav', 'Jason Holder',
        'Sai Sudharsan', 'Shahrukh Khan', 'Kagiso Rabada', 'Mohammed Siraj',
        'Prasidh Krishna', 'Manav Suthar', 'Gurnoor Singh Brar',
        'Ishant Sharma', 'Ashok Sharma', 'Prithvi Raj Yarra', 'Luke Wood',
        'Rahul Tewatia', 'Rashid Khan',
    ],
    'LSG': [
        'Rishabh Pant', 'Aiden Markram', 'Himmat Singh', 'Matthew Breetzke',
        'Mukul Choudhary', 'Akshat Raghuwanshi', 'Josh Inglis',
        'Nicholas Pooran', 'Mitchell Marsh', 'Abdul Samad', 'Shahbaz Ahamad',
        'Arshin Kulkarni', 'Wanindu Hasaranga', 'Ayush Badoni',
        'Mohammad Shami', 'Avesh Khan', 'M. Siddharth', 'Digvesh Singh',
        'Akash Singh', 'Prince Yadav', 'Arjun Tendulkar', 'Anrich Nortje',
        'Naman Tiwari', 'Mayank Yadav', 'Mohsin Khan',
    ],
}


def normalize(name):
    """Normalize name for comparison."""
    n = name.lower().strip()
    n = n.replace('.', '').replace("'", "")
    # Common variations
    n = n.replace('surya kumar yadav', 'suryakumar yadav')
    n = n.replace('n tilak varma', 'tilak varma')
    n = n.replace('mohd arshad khan', 'arshad khan')
    n = n.replace('m siddharth', 'siddharth')
    n = n.replace('t natarajan', 'natarajan')
    n = n.replace('shahbaz ahamad', 'shahbaz ahmed')
    return n


# Load our CSV
squads = pd.read_csv(DATA / 'ipl_2026_full_squads.csv')

print("=" * 70)
print("VERIFICATION: Our CSV vs Official IPL Website")
print("=" * 70)

name_fixes = {}  # track names we need to fix in our CSV
missing_from_csv = {}
extra_in_csv = {}

for team in sorted(OFFICIAL.keys()):
    official_names = set(normalize(n) for n in OFFICIAL[team])
    csv_team = squads[squads['Team'] == team]
    csv_names = set(normalize(n) for n in csv_team['Player'])

    # What's on IPL site but not in our CSV
    missing = official_names - csv_names
    # What's in our CSV but not on IPL site
    extra = csv_names - official_names

    print(f"\n{team} ({len(OFFICIAL[team])} official, {len(csv_team)} CSV):")

    if not missing and not extra:
        print(f"  ✅ Perfect match!")
    else:
        if missing:
            # Try to find close matches between missing and extra
            for m in sorted(missing):
                matched = False
                for e in sorted(extra):
                    # Check if they might be the same person
                    if (m.split()[-1] == e.split()[-1] or
                            m.split()[0] == e.split()[0]):
                        print(f"  🔄 NAME MISMATCH: Official='{m}' vs CSV='{e}'")
                        # Find the original names
                        orig_official = [n for n in OFFICIAL[team] if normalize(n) == m][0]
                        orig_csv = [n for n in csv_team['Player'] if normalize(n) == e][0]
                        name_fixes[orig_csv] = orig_official
                        matched = True
                        extra.discard(e)
                        break
                if not matched:
                    orig = [n for n in OFFICIAL[team] if normalize(n) == m][0]
                    print(f"  ❌ MISSING from CSV: {orig}")
                    missing_from_csv.setdefault(team, []).append(orig)

            for e in sorted(extra):
                orig = [n for n in csv_team['Player'] if normalize(n) == e]
                if orig:
                    print(f"  ➕ EXTRA in CSV (not on IPL site): {orig[0]}")
                    extra_in_csv.setdefault(team, []).append(orig[0])

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Name fixes needed: {len(name_fixes)}")
for old, new in name_fixes.items():
    print(f"  '{old}' → '{new}'")
print(f"Missing from CSV: {sum(len(v) for v in missing_from_csv.values())}")
for team, players in missing_from_csv.items():
    for p in players:
        print(f"  {team}: {p}")
print(f"Extra in CSV: {sum(len(v) for v in extra_in_csv.values())}")
for team, players in extra_in_csv.items():
    for p in players:
        print(f"  {team}: {p}")

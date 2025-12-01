import re
from pathlib import Path
import pandas as pd

def generate_target_table(project_root, csv_filename, sort_col, display_cols, title, limit=5):
    """
    Reads a valued CSV, sorts it, picks top `limit` (or all if limit is None), and returns an HTML table string.
    """
    try:
        # Look for valued file first
        valued_path = project_root / 'results' / 'analysis' / 'strategic' / 'valued' / f"valued_{csv_filename}"
        if not valued_path.exists():
            # Fallback to raw file
            raw_path = project_root / 'results' / 'analysis' / 'strategic' / csv_filename
            if not raw_path.exists():
                return f"<p><em>Data not found for {title}</em></p>"
            df = pd.read_csv(raw_path)
        else:
            df = pd.read_csv(valued_path)
            
        # Sort and Head
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False)
        
        if limit is not None:
            df = df.head(limit)
            
        # Select Columns
        # Filter display_cols that actually exist
        cols_to_show = [c for c in display_cols if c in df.columns]
        
        # Add Player Name if it's the index or a column
        if 'Player' not in cols_to_show:
            if 'Player' in df.columns:
                cols_to_show.insert(0, 'Player')
            elif 'batter_name' in df.columns:
                cols_to_show.insert(0, 'batter_name')
            elif 'bowler_name' in df.columns:
                cols_to_show.insert(0, 'bowler_name')
            else:
                # If index is player name (often the case in these CSVs if saved with index)
                # Check if first column is unnamed (index)
                if df.columns[0].startswith('Unnamed'):
                    df = df.rename(columns={df.columns[0]: 'Player'})
                    cols_to_show.insert(0, 'Player')
        
        final_df = df[cols_to_show].copy()
        
        # Formatting
        # Round numeric columns
        for col in final_df.select_dtypes(include=['float', 'int']).columns:
            final_df[col] = final_df[col].apply(lambda x: f"{x:.2f}")
            
        html = f"<h4>{title} (Top {limit if limit else 'All'})</h4>"
        html += final_df.to_html(index=False, classes='table', border=0)
        return html
        
    except Exception as e:
        return f"<p><em>Error generating table for {title}: {e}</em></p>"

def main():
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / 'results' / 'analysis' / 'strategic' / 'dc_strategy_blog.html'
    
    print("Generating blog...")
    
    # --- Augment Tables with Watchlist ---
    watchlist_path = project_root / 'results' / 'analysis' / 'strategic' / 'valued' / 'valued_watchlist.csv'
    if watchlist_path.exists():
        df_watch = pd.read_csv(watchlist_path)
        
        # Define Mappings
        # Openers: Seifert, Taide
        openers_add = df_watch[df_watch['User_Name'].isin(['Tim Seifert', 'Atharva Taide'])].copy()
        openers_add = openers_add.rename(columns={
            'PP_SR': 'SR_powerplay', 'PP_RAA': 'RAA_powerplay', 'WAR_2026': 'WAR_2026_Proj'
        })
        
        # Middle Order: Manohar
        middle_add = df_watch[df_watch['User_Name'].isin(['Abhinav Manohar'])].copy()
        middle_add = middle_add.rename(columns={
            'Middle_RAA': 'RAA_middle', 'WAR_2026': 'WAR_2026_Proj'
        })
        # Note: SR_middle might be missing in watchlist, let's fill with 0 or N/A if needed, 
        # but generate_target_table handles missing cols by just not showing them if not in df? 
        # No, it selects cols_to_show. We need to ensure cols exist.
        middle_add['SR_middle'] = 0 # Placeholder if missing
        
        # Pacers: Johnson, Sakariya, Madhwal, Naveen
        pacers_add = df_watch[df_watch['User_Name'].isin(['Spencer Johnson', 'Chetan Sakariya', 'Akash Madhwal', 'Naveen-ul-Haq'])].copy()
        pacers_add = pacers_add.rename(columns={
            'PP_Wickets': 'Wickets_powerplay', 'PP_RAA_Bowl': 'RAA_powerplay', 'WAR_2026': 'WAR_2026_Proj'
        })
        
        def save_augmented(original_file, df_add, output_name):
            path = project_root / 'results' / 'analysis' / 'strategic' / 'valued' / f"valued_{original_file}"
            if path.exists():
                df_orig = pd.read_csv(path)
                # Concat
                df_aug = pd.concat([df_orig, df_add], ignore_index=True)
                # Remove duplicates based on Player name, keeping the one from watchlist (last) or original?
                # Let's keep original if exists, but user wants to ensure they are there.
                # If they are already in top 5, we don't want duplicates.
                df_aug = df_aug.drop_duplicates(subset='Player', keep='first')
                
                # Save
                out_path = project_root / 'results' / 'analysis' / 'strategic' / 'valued' / f"valued_{output_name}"
                df_aug.to_csv(out_path, index=False)
                return output_name
            return original_file

        file_openers = save_augmented('target_openers_all.csv', openers_add, 'augmented_openers.csv')
        file_middle = save_augmented('target_middle_order.csv', middle_add, 'augmented_middle.csv')
        file_pp = save_augmented('target_pp_pacers.csv', pacers_add, 'augmented_pp.csv')
        
    else:
        file_openers = 'target_openers_all.csv'
        file_middle = 'target_middle_order.csv'
        file_pp = 'target_pp_pacers.csv'

    # --- Generate Tables ---
    # 1. Openers (Module A)
    table_openers = generate_target_table(
        project_root, file_openers, 'SR_powerplay', 
        ['SR_powerplay', 'RAA_powerplay', 'WAR_2026', 'Est_Price', 'Est_Price_VOMAM', 'Est_Price_VOPE'], "Top Explosive Openers",
        limit=None # Show all (Top + Added)
    )
    
    # 2. Middle Order (Module B)
    table_middle = generate_target_table(
        project_root, file_middle, 'RAA_middle',
        ['SR_middle', 'RAA_middle', 'WAR_2026', 'Est_Price', 'Est_Price_VOMAM', 'Est_Price_VOPE'], "Top Middle Order Targets",
        limit=None
    )
    
    # 3. PP Pacers (Module C)
    table_pp = generate_target_table(
        project_root, file_pp, 'Wickets_powerplay',
        ['Wickets_powerplay', 'RAA_powerplay', 'WAR_2026', 'Est_Price', 'Est_Price_VOMAM', 'Est_Price_VOPE'], "Top Powerplay Pacers",
        limit=None
    )
    
    # 4. Death Bowlers (Module D - New)
    # File: target_death_bowlers.csv
    table_death = generate_target_table(
        project_root, 'target_death_bowlers.csv', 'RAA',
        ['Econ', 'Dot_Pct', 'Wickets', 'RAA', 'WAR_2026', 'Est_Price', 'Est_Price_VOMAM', 'Est_Price_VOPE'], "Top Death Bowlers"
    )


    # --- HTML Template ---
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DC Strategy 2026</title>
    <style>
        :root {{
            --primary: #0078BC; /* DC Blue */
            --secondary: #EF1B23; /* DC Red */
            --bg: #121212;
            --text: #e0e0e0;
            --card-bg: #1e1e1e;
        }}
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: var(--primary);
            margin-top: 1.5em;
        }}
        h1 {{
            font-size: 2.5em;
            border-bottom: 2px solid var(--secondary);
            padding-bottom: 10px;
            text-align: center;
        }}
        h2 {{
            border-left: 5px solid var(--secondary);
            padding-left: 15px;
            background: linear-gradient(90deg, rgba(239,27,35,0.1) 0%, rgba(0,0,0,0) 100%);
        }}
        .plot-container {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
            background: white;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--card-bg);
            font-size: 0.9em;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background-color: var(--primary);
            color: white;
        }}
        tr:hover {{ background-color: #2a2a2a; }}
        .insight-box {{
            border-left: 4px solid var(--secondary);
            background: rgba(239, 27, 35, 0.1);
            padding: 15px;
            margin: 15px 0;
        }}
        .impact-badge {{
            background-color: var(--secondary);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .budget-box {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #444;
        }}
        .budget-box h4 {{ margin-top: 0; color: var(--secondary); }}
        .budget-list {{ list-style: none; padding: 0; }}
        .budget-list li {{ margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }}
        .budget-list span {{ float: right; font-weight: bold; color: #aaa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Delhi Capitals: The Road to Redemption (2026 Strategy)</h1>

        <h2>Phase 1: The Diagnosis</h2>
        <p>In 2025, DC lost games in two specific phases: The first 6 overs (with the ball) and the middle overs (with the bat).</p>

        <h3>1. The Powerplay Bowling Crisis</h3>
        <p>We failed to break opening partnerships. The data shows DC ranked near the bottom for Powerplay Wickets.</p>
        <div class="plot-container"><iframe src="diagnosis_pp_bowling.html" loading="lazy"></iframe></div>
        <div class="insight-box">
            <strong>The Fix:</strong> We don't just need economy; we need <em>wickets</em>. A swing bowler is non-negotiable.
        </div>

        <h3>2. The Middle-Over Slump</h3>
        <p>Our middle order (overs 7-15) was too slow compared to playoff teams.</p>
        <div class="plot-container"><iframe src="diagnosis_middle_overs.html" loading="lazy"></iframe></div>
        
        <h3>3. The Death Bowling Leak</h3>
        <p>We leaked too many runs at the death.</p>
        <div class="plot-container"><iframe src="diagnosis_death_bowling.html" loading="lazy"></iframe></div>

        <h3>4. The Opening Instability</h3>
        <p>DC struggled to find a consistent opening pair. We used <strong>7 different opening combinations</strong> in 2025, significantly higher than the league average of <strong>2.8</strong>. This constant chopping and changing prevented any rhythm at the top.</p>
        <div class="plot-container"><iframe src="diagnosis_opening_stability.html" loading="lazy"></iframe></div>

        <hr />

        <h2>Phase 2: The Data-Backed Solutions</h2>

        <h3>Module A: Fixing the Opening Instability</h3>
        <p><strong>Problem:</strong> Unstable starts put too much pressure on the middle order.<br>
        <strong>Solution:</strong> Pair an anchor (KL Rahul) with a statistical outlier who maximizes the field restrictions.</p>
        <div class="plot-container"><iframe src="solution_openers_all.html" loading="lazy"></iframe></div>
        
        {table_openers}

        <h3>Module B: The Middle Order Engine</h3>
        <p><strong>Problem:</strong> Slow scoring against spin in overs 7-15.<br>
        <strong>Solution:</strong> Target high-RAA middle order batters who can dominate spin.</p>
        <div class="plot-container"><iframe src="solution_middle_order.html" loading="lazy"></iframe></div>
        
        {table_middle}

        <h3>Module C: The Powerplay Pacers</h3>
        <p><strong>Problem:</strong> Lack of early wickets.<br>
        <strong>Solution:</strong> Target bowlers who strike early.</p>
        <div class="plot-container"><iframe src="solution_pp_pacers.html" loading="lazy"></iframe></div>
        
        {table_pp}
        
        <h3>Module D: The Death Bowlers</h3>
        <p><strong>Problem:</strong> Leaking runs at the end.<br>
        <strong>Solution:</strong> Target high dot-ball percentage bowlers.</p>
        <div class="plot-container"><iframe src="solution_death_bowlers.html" loading="lazy"></iframe></div>
        
        {table_death}

        <hr />
        
        <h2>Phase 3: Strategic Targets & Budget</h2>
        
        <div class="budget-box">
            <h4>Primary Targets (₹18-20 Cr)</h4>
            <ul class="budget-list">
                <li>Explosive overseas opener (Seifert/Inglis) + JFM depth <span>~₹8 Cr</span></li>
                <li>Overseas fast bowler (Spencer Johnson type) <span>~₹4 Cr</span></li>
                <li>Young Indian pacer (Sakaria/Madhavan) <span>~₹2 Cr</span></li>
                <li>Experienced middle-order bat <span>~₹4 Cr</span></li>
                <li>Buffer for bidding wars <span>~₹2 Cr</span></li>
            </ul>
            <h4>Value Picks (₹1.5 Cr remaining)</h4>
            <ul class="budget-list">
                <li>Indian floater batter (Manohar/Taide) <span>Base Price</span></li>
                <li>Uncapped spinner <span>Base Price</span></li>
                <li>Best scouting pick <span>Base Price</span></li>
            </ul>
        </div>
        
            </ul>
        </div>
        
        <hr />
        
        <h2>Phase 4: The Vision (12-Man Impact Squad)</h2>
        <p>Leveraging the Impact Player rule, here is our tactical XI + 1:</p>
        <table>
        <thead>
        <tr>
        <th style="text-align: left;">Order</th>
        <th style="text-align: left;">Player</th>
        <th style="text-align: left;">Role / Rationale</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td>1</td>
        <td>🇮🇳 KL Rahul (C)</td>
        <td>Anchor & Captain</td>
        </tr>
        <tr>
        <td>2</td>
        <td>✈️ <strong>Overseas Buy</strong></td>
        <td><em>Fraser-McGurk / Maxwell / Bairstow</em></td>
        </tr>
        <tr>
        <td>3</td>
        <td>🇮🇳 <strong>Nitish Rana</strong></td>
        <td><em>Stabilizer & Spin Basher</em></td>
        </tr>
        <tr>
        <td>4</td>
        <td>🇮🇳 Karun Nair</td>
        <td>Technical Stability</td>
        </tr>
        <tr>
        <td>5</td>
        <td>🇿🇦 Tristan Stubbs</td>
        <td>Retained Finisher</td>
        </tr>
        <tr>
        <td>6</td>
        <td>🇮🇳 Axar Patel</td>
        <td>Retained All-Rounder</td>
        </tr>
        <tr>
        <td>7</td>
        <td>🇮🇳 Ashutosh Sharma</td>
        <td>Death Overs Hitting</td>
        </tr>
        <tr>
        <td>8</td>
        <td>🇮🇳 Vipraj Nigam</td>
        <td>Mystery Spin / X-Factor</td>
        </tr>
        <tr>
        <td>9</td>
        <td>🇮🇳 Kuldeep Yadav</td>
        <td>Retained Wicket-Taker</td>
        </tr>
        <tr>
        <td>10</td>
        <td>✈️ <strong>Mitchell Starc</strong></td>
        <td><em>Marquee Target (Death Specialist)</em></td>
        </tr>
        <tr>
        <td>11</td>
        <td>🇮🇳 T. Natarajan</td>
        <td><em>Return from Injury (Yorker King)</em></td>
        </tr>
        <tr style="background-color: #2a2a2a; border-top: 2px solid #EF1B23;">
        <td><strong>12</strong></td>
        <td>✈️ <strong>Overseas Pacer</strong></td>
        <td><span class="impact-badge">IMPACT</span> <em>Behrendorff / Nortje (PP Specialist)</em></td>
        </tr>
        </tbody>
        </table>
        
        <p><strong>Strategy Note:</strong> This lineup offers immense flexibility. If batting first, the Overseas Pacer comes in for Karun Nair or Ashutosh Sharma in the second innings. Nitish Rana controls the middle overs, allowing Stubbs and Ashutosh to explode at the death.</p>
    </div>
    <footer style="text-align: center; padding: 40px; color: #666;">
        <p>Generated by CricWAR Analysis Engine</p>
    </footer>
</body>
</html>
    """
    
    print(f"Writing blog to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(html_content)
    print("Done.")

if __name__ == "__main__":
    main()

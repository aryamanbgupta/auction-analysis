import pandas as pd
from pathlib import Path
import markdown

def generate_report():
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'results' / 'analysis' / 'moneyball' / 'moneyball_targets.csv'
    output_md = project_root / 'results' / 'analysis' / 'moneyball' / 'moneyball_report.md'
    output_html = project_root / 'results' / 'analysis' / 'moneyball' / 'moneyball_report.html'
    
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Formatting
    df['WAR_2026'] = df['WAR_2026'].round(2)
    df['Predicted_Price_Cr'] = df['Predicted_Price_Cr'].round(2)
    df['Value_Ratio'] = df['Value_Ratio'].round(2)
    
    md = "# 💰 Moneyball IPL 2026: The Value Draft\n\n"
    md += "### Objective\n"
    md += "Identify the most undervalued players in the auction pool using the **VOMAM (Value Over Market Adjusted Model)**. "
    md += "We predict the *Market Price* based on 2026 WAR projections and compare it to the *Actual Value* (WAR).\n\n"
    
    md += "> **Metric:** `Value Ratio = Projected WAR / Predicted Price`\n"
    md += "> * Higher is better. A ratio > 1.0 means you get more WAR per Crore than the market average.\n\n"
    
    # Top 10 Value Buys
    md += "## 💎 Top 10 Value Buys\n"
    md += "These players offer the highest return on investment.\n\n"
    top_10 = df.head(10)[['name', 'role', 'country', 'WAR_2026', 'Predicted_Price_Cr', 'Value_Ratio']]
    md += top_10.to_markdown(index=False) + "\n\n"
    
    # Role-wise Analysis
    md += "## 🎯 Role-wise Value Picks\n\n"
    
    roles = df['role'].unique()
    for role in roles:
        role_df = df[df['role'] == role].head(5)
        if not role_df.empty:
            md += f"### Best Value: {role}\n"
            md += role_df[['name', 'country', 'WAR_2026', 'Predicted_Price_Cr', 'Value_Ratio']].to_markdown(index=False) + "\n\n"
            
    # Overpriced Traps (Bottom 5)
    md += "## ⚠️ Overpriced Traps (Lowest Value)\n"
    md += "These players are predicted to be expensive relative to their WAR contribution.\n\n"
    bottom_5 = df.tail(5).sort_values('Value_Ratio', ascending=True)[['name', 'role', 'WAR_2026', 'Predicted_Price_Cr', 'Value_Ratio']]
    md += bottom_5.to_markdown(index=False) + "\n\n"
    
    # Save Markdown
    with open(output_md, 'w') as f:
        f.write(md)
    print(f"Markdown report saved to {output_md}")
    
    # Convert to HTML
    html_content = markdown.markdown(md, extensions=['tables'])
    
    # Add CSS (Dark Mode)
    css = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #121212; color: #e0e0e0; line-height: 1.6; padding: 20px; max-width: 1000px; margin: 0 auto; }
        h1, h2, h3 { color: #2563eb; border-bottom: 1px solid #333; padding-bottom: 10px; }
        h1 { text-align: center; color: #60a5fa; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background-color: #1e1e1e; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #333; }
        th { background-color: #2563eb; color: white; }
        tr:hover { background-color: #2d2d2d; }
        blockquote { border-left: 4px solid #2563eb; padding-left: 15px; color: #9ca3af; margin: 20px 0; }
        code { background-color: #333; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
    </style>
    """
    
    full_html = f"<!DOCTYPE html><html><head><title>Moneyball Analysis</title>{css}</head><body>{html_content}</body></html>"
    
    with open(output_html, 'w') as f:
        f.write(full_html)
    print(f"HTML report saved to {output_html}")

if __name__ == "__main__":
    generate_report()

import re
from pathlib import Path
import markdown
import pandas as pd

def main():
    project_root = Path(__file__).parent.parent.parent
    report_path = project_root / 'scripts' / 'analysis' / 'dc_strategic_report.md'
    output_path = project_root / 'results' / 'analysis' / 'strategic' / 'dc_strategy_blog.html'
    
    print(f"Reading report from {report_path}...")
    with open(report_path, 'r') as f:
        md_content = f.read()
    
    # Custom processing to replace Image + Link blocks with Iframes
    # Pattern: ![Alt](...png) \n * [Interactive Graph](...html)
    # We want to extract the HTML filename.
    
    # Regex to find the block
    # Note: The report has relative paths like ../../results/analysis/strategic/filename.html
    # We want to extract just 'filename.html' because the blog will be in the same dir.
    
    def iframe_replacement(match):
        # match.group(0) is the whole block
        # We need to find the .html link
        html_match = re.search(r'\((.*?\.html)\)', match.group(0))
        if html_match:
            full_path = html_match.group(1)
            filename = Path(full_path).name
            return f'<div class="plot-container"><iframe src="{filename}" loading="lazy"></iframe></div>'
        return match.group(0)

    # Regex covers: Image line + optional newline + bullet point with link
    # ![...](...)
    # * [Interactive Graph](...)
    pattern = r'!\[.*?\]\(.*?\)\s*\*\s*\[Interactive Graph\]\(.*?\)'
    
    # Pre-process markdown to replace these blocks with placeholders or direct HTML
    # But `markdown` lib might escape HTML.
    # Better approach: Convert MD to HTML first, then replace the specific HTML structure?
    # Or replace in MD with a placeholder, convert, then replace placeholder.
    
    # Let's try replacing in MD with raw HTML (Markdown usually supports raw HTML)
    processed_md = re.sub(pattern, iframe_replacement, md_content, flags=re.DOTALL)
    
    # CSV Table Replacement
    # Pattern: [View Valued Targets Table](...csv)
    def table_replacement(match):
        csv_path = match.group(1)
        # Resolve path relative to report location?
        # Report is in scripts/analysis/
        # Link is ../../results/...
        # We need absolute path to read it
        abs_path = (project_root / 'scripts' / 'analysis' / csv_path).resolve()
        
        if not abs_path.exists():
            return f"<p><em>Table not found: {csv_path}</em></p>"
            
        try:
            df = pd.read_csv(abs_path)
            # Convert to HTML table
            html_table = df.to_html(index=False, classes='table table-striped', float_format=lambda x: '{:.2f}'.format(x))
            return f'<div class="table-container">{html_table}</div>'
        except Exception as e:
            return f"<p><em>Error loading table: {e}</em></p>"

    csv_pattern = r'\[View Valued Targets Table\]\((.*?\.csv)\)'
    processed_md = re.sub(csv_pattern, table_replacement, processed_md)
    
    # Convert to HTML
    html_body = markdown.markdown(processed_md, extensions=['tables', 'fenced_code'])
    
    # HTML Template
    html_template = f"""
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
        a {{ color: var(--secondary); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
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
            border-radius: 4px;
            background: white; /* Plotly plots are white by default */
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--card-bg);
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
        
        blockquote {{
            border-left: 4px solid var(--secondary);
            margin: 0;
            padding-left: 20px;
            color: #aaa;
            font-style: italic;
        }}
        
        /* Comparison Legend */
        .legend {{
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }}
        .legend span {{ margin: 0 10px; }}
        .blue {{ color: #0078BC; }}
        .red {{ color: #EF1B23; }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
    <footer style="text-align: center; padding: 40px; color: #666;">
        <p>Generated by CricWAR Analysis Engine</p>
    </footer>
</body>
</html>
    """
    
    print(f"Writing blog to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(html_template)
    print("Done.")

if __name__ == "__main__":
    main()

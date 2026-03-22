"""
Generate an Interactive HTML Table for 2026 IPL Auction Analytics.

Creates a filterable, sortable table using DataTables.js
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def generate_interactive_table():
    """Generate an interactive HTML table from auction data."""
    print("Loading auction data...")
    input_path = PROJECT_ROOT / 'results' / 'analysis' / 'auction_2026_comprehensive.csv'
    
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return
        
    df = pd.read_csv(input_path)
    
    # 1. Force numeric conversion for key columns to handle any string leftovers
    # This ensures "30" becomes 30 (int/float) and not string "30"
    numeric_columns = [
        'base_price_lakh', 'war_2026', 'age', 
        'vomam_price_cr', 'vope_price_cr', 'avg_price_cr'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify ALL numeric columns (including generated stats)
    all_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Round numeric columns for cleaner display
    for col in all_numeric_cols:
        df[col] = df[col].round(2)
    
    # Replace NaN with None for valid JSON nulls
    # DataTables handles `null` gracefully in sorting/filtering
    df = df.where(pd.notnull(df), None)
    
    # Convert to records for JSON
    data = df.to_dict(orient='records')
    columns = df.columns.tolist()
    
    # Create column definitions for DataTables
    column_defs = []
    for col in columns:
        col_def = {
            'data': col,
            'title': col.replace('_', ' ').title(),
            'defaultContent': ''
        }
        
        # Set type for numeric columns to ensure correct sorting
        if col in all_numeric_cols:
            col_def['type'] = 'num'
            col_def['className'] = 'dt-body-right'
            
        column_defs.append(col_def)
    
    # Get unique values for categorical filters
    roles = sorted([r for r in df['role'].dropna().unique() if r])
    countries = sorted([c for c in df['country'].dropna().unique() if c])
    capped_values = sorted([c for c in df['capped'].dropna().unique() if c])
    war_sources = sorted([s for s in df['war_source'].dropna().unique() if s])
    
    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2026 IPL Auction Analytics</title>
    
    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/fixedheader/3.4.0/css/fixedHeader.dataTables.min.css">
    
    <style>
        :root {{
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --positive: #22c55e;
            --negative: #ef4444;
            --warning: #f59e0b;
        }}
        
        * {{ box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{ max-width: 100%; margin: 0 auto; }}
        
        header {{
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, var(--bg-card) 0%, #2d3a4f 100%);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        h1 {{
            margin: 0 0 10px 0;
            font-size: 2.2rem;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{ color: var(--text-secondary); font-size: 1rem; }}
        
        .stats-row {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .stat-card {{
            background: var(--bg-dark);
            padding: 12px 20px;
            border-radius: 10px;
            text-align: center;
            min-width: 100px;
        }}
        
        .stat-value {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
        .stat-label {{ font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; }}
        
        /* Filter Panel */
        .filter-panel {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .filter-panel h3 {{
            margin: 0 0 15px 0;
            font-size: 1rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .filter-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .filter-group label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .filter-group input,
        .filter-group select {{
            background: var(--bg-dark);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 10px 12px;
            color: var(--text-primary);
            font-size: 0.9rem;
        }}
        
        .filter-group input:focus,
        .filter-group select:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }}
        
        .range-inputs {{
            display: flex;
            gap: 8px;
        }}
        
        .range-inputs input {{
            flex: 1;
            min-width: 0;
        }}
        
        .filter-actions {{
            display: flex;
            gap: 10px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s;
        }}
        
        .btn-primary {{
            background: var(--accent);
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #2563eb;
            transform: translateY(-1px);
        }}
        
        .btn-secondary {{
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .btn-secondary:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        
        .active-filters {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        
        .filter-tag {{
            background: var(--accent);
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .filter-tag .remove {{
            cursor: pointer;
            opacity: 0.7;
        }}
        
        .filter-tag .remove:hover {{ opacity: 1; }}
        
        /* Controls */
        .controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .column-toggle {{
            background: var(--bg-card);
            padding: 12px 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .column-toggle h3 {{
            margin: 0 0 8px 0;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        
        .toggle-group {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        
        .toggle-btn {{
            padding: 6px 14px;
            border: 1px solid rgba(255,255,255,0.2);
            background: transparent;
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }}
        
        .toggle-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
        .toggle-btn.active {{ background: var(--accent); border-color: var(--accent); color: white; }}
        
        .table-container {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 15px;
            overflow-x: auto;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        /* DataTables Styling */
        table.dataTable {{ border-collapse: collapse !important; width: 100% !important; }}
        
        table.dataTable thead th {{
            background: var(--bg-dark) !important;
            color: var(--text-secondary) !important;
            padding: 10px 8px !important;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--accent) !important;
            white-space: nowrap;
        }}
        
        table.dataTable tbody td {{
            padding: 8px !important;
            border-bottom: 1px solid rgba(255,255,255,0.05) !important;
            color: var(--text-primary);
            font-size: 0.85rem;
        }}
        
        table.dataTable tbody tr {{ background: transparent !important; }}
        table.dataTable tbody tr:hover {{ background: rgba(59, 130, 246, 0.1) !important; }}
        
        .dataTables_wrapper .dataTables_filter input {{
            background: var(--bg-dark);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 8px 12px;
            color: var(--text-primary);
            margin-left: 10px;
        }}
        
        .dataTables_wrapper .dataTables_length select {{
            background: var(--bg-dark);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 8px;
            color: var(--text-primary);
        }}
        
        .dataTables_wrapper .dataTables_filter label,
        .dataTables_wrapper .dataTables_length label {{ color: var(--text-secondary); }}
        .dataTables_wrapper .dataTables_info {{ color: var(--text-secondary); }}
        
        .dataTables_wrapper .dataTables_paginate .paginate_button {{
            color: var(--text-secondary) !important;
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 6px !important;
            margin: 0 2px !important;
        }}
        
        .dataTables_wrapper .dataTables_paginate .paginate_button:hover,
        .dataTables_wrapper .dataTables_paginate .paginate_button.current {{
            background: var(--accent) !important;
            color: white !important;
            border-color: var(--accent) !important;
        }}
        
        .dt-buttons {{ margin-bottom: 10px; }}
        
        .dt-button {{
            background: var(--bg-dark) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 6px !important;
            color: var(--text-primary) !important;
            padding: 6px 12px !important;
            margin-right: 6px !important;
        }}
        
        .dt-button:hover {{ background: var(--accent) !important; border-color: var(--accent) !important; }}
        
        .positive {{ color: var(--positive); font-weight: 600; }}
        .negative {{ color: var(--negative); font-weight: 600; }}
        
        .results-count {{
            background: var(--warning);
            color: #000;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        /* Alignment */
        .dt-body-right {{ text-align: right; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏏 2026 IPL Auction Analytics</h1>
            <p class="subtitle">Filter, sort, and analyze 350 auction players</p>
            
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-value" id="visibleCount">{len(df)}</div>
                    <div class="stat-label">Visible</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(df)}</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(columns)}</div>
                    <div class="stat-label">Columns</div>
                </div>
            </div>
        </header>
        
        <!-- Filter Panel -->
        <div class="filter-panel">
            <h3>🔍 Filters <span id="activeFilterCount" class="results-count" style="display:none;">0 active</span></h3>
            
            <div class="filter-grid">
                <!-- Categorical Filters -->
                <div class="filter-group">
                    <label>Role</label>
                    <select id="filter-role">
                        <option value="">All Roles</option>
                        {chr(10).join(f'<option value="{r}">{r}</option>' for r in roles)}
                    </select>
                </div>
                
                <div class="filter-group">
                    <label>Country</label>
                    <select id="filter-country">
                        <option value="">All Countries</option>
                        {chr(10).join(f'<option value="{c}">{c}</option>' for c in countries)}
                    </select>
                </div>
                
                <div class="filter-group">
                    <label>Capped Status</label>
                    <select id="filter-capped">
                        <option value="">All</option>
                        {chr(10).join(f'<option value="{c}">{c}</option>' for c in capped_values)}
                    </select>
                </div>
                
                <div class="filter-group">
                    <label>WAR Source</label>
                    <select id="filter-war-source">
                        <option value="">All Sources</option>
                        {chr(10).join(f'<option value="{s}">{s}</option>' for s in war_sources)}
                    </select>
                </div>
                
                <!-- Numeric Range Filters -->
                <div class="filter-group">
                    <label>Base Price (Lakh)</label>
                    <div class="range-inputs">
                        <input type="number" id="filter-price-min" placeholder="Min" step="10">
                        <input type="number" id="filter-price-max" placeholder="Max" step="10">
                    </div>
                </div>
                
                <div class="filter-group">
                    <label>WAR 2026</label>
                    <div class="range-inputs">
                        <input type="number" id="filter-war-min" placeholder="Min" step="0.1">
                        <input type="number" id="filter-war-max" placeholder="Max" step="0.1">
                    </div>
                </div>
                
                <div class="filter-group">
                    <label>Age</label>
                    <div class="range-inputs">
                        <input type="number" id="filter-age-min" placeholder="Min">
                        <input type="number" id="filter-age-max" placeholder="Max">
                    </div>
                </div>
                
                <div class="filter-group">
                    <label>Avg Price (Cr)</label>
                    <div class="range-inputs">
                        <input type="number" id="filter-avgprice-min" placeholder="Min" step="0.5">
                        <input type="number" id="filter-avgprice-max" placeholder="Max" step="0.5">
                    </div>
                </div>
            </div>
            
            <div class="filter-actions">
                <button class="btn btn-primary" onclick="applyFilters()">Apply Filters</button>
                <button class="btn btn-secondary" onclick="clearFilters()">Clear All</button>
            </div>
            
            <div class="active-filters" id="activeFilters"></div>
        </div>
        
        <div class="controls">
            <div class="column-toggle">
                <h3>Column Groups</h3>
                <div class="toggle-group">
                    <button class="toggle-btn active" onclick="showColumnGroup('core', this)">Core Info</button>
                    <button class="toggle-btn" onclick="showColumnGroup('bat_ipl', this)">Batting (IPL)</button>
                    <button class="toggle-btn" onclick="showColumnGroup('bat_all', this)">Batting (All)</button>
                    <button class="toggle-btn" onclick="showColumnGroup('bowl_ipl', this)">Bowling (IPL)</button>
                    <button class="toggle-btn" onclick="showColumnGroup('bowl_all', this)">Bowling (All)</button>
                    <button class="toggle-btn" onclick="showColumnGroup('all', this)">Show All</button>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <table id="auctionTable" class="display nowrap" style="width:100%"></table>
        </div>
    </div>
    
    <!-- Scripts -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.colVis.min.js"></script>
    <script src="https://cdn.datatables.net/fixedheader/3.4.0/js/dataTables.fixedHeader.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    
    <script>
        // Data injected from Python
        const tableData = {json.dumps(data)};
        const columns = {json.dumps(column_defs)};
        
        const columnGroups = {{
            core: ['sr_no', 'player', 'country', 'role', 'base_price_lakh', 'capped', 'age', 'war_2026', 'war_source', 'vomam_price_cr', 'vope_price_cr', 'avg_price_cr'],
            bat_ipl: ['player', 'role', 'war_2026', 'bat_runs_pp_ipl', 'bat_balls_pp_ipl', 'bat_sr_pp_ipl', 'bat_raa_pp_ipl', 'bat_boundary_pct_pp_ipl', 'bat_sr_mid_ipl', 'bat_raa_mid_ipl', 'bat_sr_death_ipl', 'bat_raa_death_ipl'],
            bat_all: ['player', 'role', 'war_2026', 'bat_runs_pp_all', 'bat_balls_pp_all', 'bat_sr_pp_all', 'bat_raa_pp_all', 'bat_boundary_pct_pp_all', 'bat_sr_mid_all', 'bat_raa_mid_all', 'bat_sr_death_all', 'bat_raa_death_all'],
            bowl_ipl: ['player', 'role', 'war_2026', 'bowl_balls_pp_ipl', 'bowl_wkts_pp_ipl', 'bowl_econ_pp_ipl', 'bowl_raa_pp_ipl', 'bowl_econ_mid_ipl', 'bowl_raa_mid_ipl', 'bowl_econ_death_ipl', 'bowl_wkts_death_ipl', 'bowl_raa_death_ipl'],
            bowl_all: ['player', 'role', 'war_2026', 'bowl_balls_pp_all', 'bowl_wkts_pp_all', 'bowl_econ_pp_all', 'bowl_raa_pp_all', 'bowl_econ_mid_all', 'bowl_raa_mid_all', 'bowl_econ_death_all', 'bowl_wkts_death_all', 'bowl_raa_death_all'],
            all: columns.map(c => c.data)
        }};
        
        let table;
        let activeFilters = {{}};
        
        $(document).ready(function() {{
            // Custom filtering function using rowData (4th argument)
            $.fn.dataTable.ext.search.push(function(settings, data, dataIndex, rowData, counter) {{
                // rowData contains the original data object with correct types
                // We access properties directly from the JSON object
                
                // Role filter
                if (activeFilters.role && rowData.role !== activeFilters.role) return false;
                
                // Country filter
                if (activeFilters.country && rowData.country !== activeFilters.country) return false;
                
                // Capped filter
                if (activeFilters.capped && rowData.capped !== activeFilters.capped) return false;
                
                // WAR Source filter
                if (activeFilters.warSource && rowData.war_source !== activeFilters.warSource) return false;
                
                // Base Price range
                // Use parseFloat to be safe, though Python ensured they are numbers
                const price = parseFloat(rowData.base_price_lakh) || 0;
                if (activeFilters.priceMin !== null && price < activeFilters.priceMin) return false;
                if (activeFilters.priceMax !== null && price > activeFilters.priceMax) return false;
                
                // WAR range
                const war = parseFloat(rowData.war_2026) || 0;
                if (activeFilters.warMin !== null && war < activeFilters.warMin) return false;
                if (activeFilters.warMax !== null && war > activeFilters.warMax) return false;
                
                // Age range
                const age = parseFloat(rowData.age) || 0;
                if (activeFilters.ageMin !== null && age < activeFilters.ageMin) return false;
                if (activeFilters.ageMax !== null && age > activeFilters.ageMax) return false;
                
                // Avg Price range
                const avgPrice = parseFloat(rowData.avg_price_cr) || 0;
                if (activeFilters.avgPriceMin !== null && avgPrice < activeFilters.avgPriceMin) return false;
                if (activeFilters.avgPriceMax !== null && avgPrice > activeFilters.avgPriceMax) return false;
                
                return true;
            }});
            
            table = $('#auctionTable').DataTable({{
                data: tableData,
                columns: columns,
                scrollX: true,
                scrollY: '55vh',
                scrollCollapse: true,
                paging: true,
                pageLength: 50,
                lengthMenu: [[25, 50, 100, -1], [25, 50, 100, "All"]],
                order: [[ columns.findIndex(c => c.data === 'war_2026'), 'desc' ]],
                dom: 'Blfrtip',
                buttons: [
                    'colvis',
                    {{ extend: 'csv', text: 'Export CSV', filename: 'auction_2026_filtered' }},
                    {{ extend: 'excel', text: 'Export Excel', filename: 'auction_2026_filtered' }}
                ],
                drawCallback: function() {{
                    document.getElementById('visibleCount').textContent = this.api().rows({{ search: 'applied' }}).count();
                }},
                initComplete: function() {{
                    showColumnGroup('core', document.querySelector('.toggle-btn.active'));
                }}
            }});
        }});
        
        function applyFilters() {{
            // Helper to get float or null
            const getFloat = (id) => {{
                const val = document.getElementById(id).value;
                return val === '' ? null : parseFloat(val);
            }};
            
            activeFilters = {{
                role: document.getElementById('filter-role').value,
                country: document.getElementById('filter-country').value,
                capped: document.getElementById('filter-capped').value,
                warSource: document.getElementById('filter-war-source').value,
                priceMin: getFloat('filter-price-min'),
                priceMax: getFloat('filter-price-max'),
                warMin: getFloat('filter-war-min'),
                warMax: getFloat('filter-war-max'),
                ageMin: getFloat('filter-age-min'),
                ageMax: getFloat('filter-age-max'),
                avgPriceMin: getFloat('filter-avgprice-min'),
                avgPriceMax: getFloat('filter-avgprice-max')
            }};
            
            table.draw();
            updateFilterTags();
        }}
        
        function clearFilters() {{
            document.getElementById('filter-role').value = '';
            document.getElementById('filter-country').value = '';
            document.getElementById('filter-capped').value = '';
            document.getElementById('filter-war-source').value = '';
            document.getElementById('filter-price-min').value = '';
            document.getElementById('filter-price-max').value = '';
            document.getElementById('filter-war-min').value = '';
            document.getElementById('filter-war-max').value = '';
            document.getElementById('filter-age-min').value = '';
            document.getElementById('filter-age-max').value = '';
            document.getElementById('filter-avgprice-min').value = '';
            document.getElementById('filter-avgprice-max').value = '';
            
            activeFilters = {{}};
            table.draw();
            updateFilterTags();
        }}
        
        function updateFilterTags() {{
            const container = document.getElementById('activeFilters');
            const countBadge = document.getElementById('activeFilterCount');
            let tags = [];
            
            if (activeFilters.role) tags.push(`Role: ${{activeFilters.role}}`);
            if (activeFilters.country) tags.push(`Country: ${{activeFilters.country}}`);
            if (activeFilters.capped) tags.push(`Capped: ${{activeFilters.capped}}`);
            if (activeFilters.warSource) tags.push(`Source: ${{activeFilters.warSource}}`);
            if (activeFilters.priceMin !== null) tags.push(`Base Price ≥ ${{activeFilters.priceMin}}L`);
            if (activeFilters.priceMax !== null) tags.push(`Base Price ≤ ${{activeFilters.priceMax}}L`);
            if (activeFilters.warMin !== null) tags.push(`WAR ≥ ${{activeFilters.warMin}}`);
            if (activeFilters.warMax !== null) tags.push(`WAR ≤ ${{activeFilters.warMax}}`);
            if (activeFilters.ageMin !== null) tags.push(`Age ≥ ${{activeFilters.ageMin}}`);
            if (activeFilters.ageMax !== null) tags.push(`Age ≤ ${{activeFilters.ageMax}}`);
            if (activeFilters.avgPriceMin !== null) tags.push(`Avg Price ≥ ${{activeFilters.avgPriceMin}} Cr`);
            if (activeFilters.avgPriceMax !== null) tags.push(`Avg Price ≤ ${{activeFilters.avgPriceMax}} Cr`);
            
            container.innerHTML = tags.map(t => `<span class="filter-tag">${{t}}</span>`).join('');
            
            if (tags.length > 0) {{
                countBadge.textContent = `${{tags.length}} active`;
                countBadge.style.display = 'inline';
            }} else {{
                countBadge.style.display = 'none';
            }}
        }}
        
        function showColumnGroup(group, btn) {{
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            
            const visibleCols = columnGroups[group];
            table.columns().every(function(index) {{
                const colName = columns[index].data;
                this.visible(visibleCols.includes(colName));
            }});
        }}
        
        // Apply filters on Enter key
        document.querySelectorAll('.filter-group input, .filter-group select').forEach(el => {{
            el.addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') applyFilters();
            }});
            el.addEventListener('change', function() {{
                if (this.tagName === 'SELECT') applyFilters();
            }});
        }});
    </script>
</body>
</html>
'''
    
    # Save HTML
    output_path = PROJECT_ROOT / 'results' / 'analysis' / 'auction_2026_interactive.html'
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✓ Generated interactive table: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_interactive_table()

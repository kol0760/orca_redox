#!/usr/bin/env python3
"""
HTML Report Generator for ORCA Redox Calculations
-------------------------------------------------
Generates beautiful, standalone HTML reports with interactive 3Dmol.js viewers,
electronic orbital energies (HOMO/LUMO/Gap), and fixed 1.24 V potential calculations.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def generate_html_report(
    summary: Dict[str, Any],
    workdir: Path,
    output_filename: str = "report.html"
) -> Path:
    mol_name = summary.get("molecule", "Molecule")
    
    # Read optimized coordinates for GN, OX, RD
    xyz_data = {}
    for st in ["GN", "OX", "RD"]:
        xyz_file = workdir / st / "01.xyz"
        if not xyz_file.exists():
            # Try finding any xyz in state dir
            candidates = list((workdir / st).glob("*.xyz")) if (workdir / st).exists() else []
            if candidates:
                xyz_file = candidates[0]
        
        if xyz_file.exists():
            with open(xyz_file, "r", encoding="utf-8", errors="ignore") as f:
                raw_xyz = f.read().strip()
                # Clean and escape for JS template literal
                xyz_data[st] = raw_xyz.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        else:
            xyz_data[st] = ""

    # Potential values (Fixed 1.24 V reference)
    e_ox_val = summary.get("E_ox_Li_V")
    e_red_val = summary.get("E_red_Li_V")
    
    e_ox_str = f"E<sub>ox</sub> = {e_ox_val:.4f} V" if e_ox_val is not None else "N/A"
    e_red_str = f"E<sub>red</sub> = {e_red_val:.4f} V" if e_red_val is not None else "N/A"

    # Electronic structure: HOMO / LUMO / Gap from GN
    homo_ev = summary.get("HOMO_eV")
    lumo_ev = summary.get("LUMO_eV")
    gap_ev = summary.get("Gap_eV")
    
    homo_str = f"{homo_ev:.4f} eV" if homo_ev is not None else "N/A"
    lumo_str = f"{lumo_ev:.4f} eV" if lumo_ev is not None else "N/A"
    gap_str = f"{gap_ev:.4f} eV" if gap_ev is not None else "N/A"

    # Build Javascript xyzData map
    js_xyz_entries = []
    for st in ["GN", "OX", "RD"]:
        if xyz_data[st]:
            js_xyz_entries.append(f'    "{st}": `{xyz_data[st]}`')
    js_xyz_data_str = "{\n" + ",\n".join(js_xyz_entries) + "\n};"

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{mol_name} - 氧化还原电位</title>
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root {{
    --primary: #1a3a6b;
    --primary-light: #3a6ab5;
    --primary-pale: #eef3fb;
    --accent: #e8b04b;
    --bg: #f4f6fa;
    --card-bg: #ffffff;
    --text: #1c2a3a;
    --text-secondary: #5a6c80;
    --text-muted: #8a9ab5;
    --border: #e2e8f0;
    --radius: 14px;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 24px rgba(26,58,107,0.10);
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: "Noto Sans SC", "Inter", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 40px 20px;
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

.container {{ max-width: 1100px; margin: 0 auto; }}

h1 {{
    text-align: center;
    color: var(--primary);
    margin-bottom: 6px;
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
}}

.subtitle {{
    text-align: center;
    color: var(--text-muted);
    margin-bottom: 36px;
    font-size: 15px;
}}

.section {{
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}}

.section h2 {{
    color: var(--primary);
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--primary-pale);
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section h2::before {{
    content: '';
    display: inline-block;
    width: 4px;
    height: 20px;
    background: var(--primary-light);
    border-radius: 2px;
}}

.structures {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}}

.structure-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}}

.structure-card:hover {{
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}}

.card-header {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    color: #fff;
    padding: 12px 18px;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.3px;
}}

.mol-viewer {{
    width: 100%;
    height: 280px;
    position: relative;
    background: #fafbfc;
}}

.style-selector {{
    padding: 10px 16px;
    font-size: 13px;
    color: var(--text-secondary);
    background: #fafbfc;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
}}

.style-selector label {{
    cursor: pointer;
    transition: color 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}

.style-selector label:hover {{ color: var(--primary); }}
.style-selector input[type="radio"] {{ accent-color: var(--primary); margin: 0; }}

.results-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}}

.result-card {{
    background: linear-gradient(135deg, var(--primary-pale) 0%, #f0f4ff 100%);
    border: 1px solid #c8d8ee;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}}

.result-card .label {{
    color: var(--text-secondary);
    font-size: 13px;
    margin-bottom: 6px;
}}

.result-card .value {{
    color: var(--primary);
    font-size: 28px;
    font-weight: 700;
    font-family: "Inter", monospace;
}}

.result-card .value sub {{ font-size: 16px; }}

.orbitals-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}}

.orbital-card {{
    background: #f8faff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}}

.orbital-card .label {{
    color: var(--text-secondary);
    font-size: 13px;
    margin-bottom: 4px;
}}

.orbital-card .value {{
    color: var(--primary);
    font-size: 22px;
    font-weight: 700;
    font-family: "Inter", monospace;
}}

table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 15px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
}}

th {{
    background: var(--primary);
    color: #fff;
    padding: 14px 18px;
    text-align: left;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.3px;
}}

td {{
    padding: 13px 18px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}}

tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: var(--primary-pale); }}

.flowchart-container {{
    margin: 8px 0 20px;
    padding: 8px 0;
}}

.flowchart-container svg {{
    width: 100%;
    height: auto;
    max-width: 100%;
}}

.method-formula {{
    margin-top: 20px;
    padding-top: 20px;
    border-top: 2px dashed #b0c8e8;
}}

.formula-box {{
    background: var(--primary-pale);
    border: 1px solid #c8d8ee;
    border-left: 4px solid var(--primary-light);
    border-radius: 10px;
    padding: 22px 28px;
    margin: 16px 0;
    font-family: "Inter", "Courier New", monospace;
    font-size: 15px;
    line-height: 2;
    color: var(--text);
}}

.formula-box p {{ margin: 4px 0; }}

.formula-box .label {{
    color: var(--text-secondary);
    font-weight: 500;
    font-family: "Noto Sans SC", "Inter", sans-serif;
}}

.formula-box .value {{
    color: var(--primary);
    font-weight: 700;
}}

.footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 24px;
    padding: 16px;
}}

@media (max-width: 768px) {{
    .structures {{ grid-template-columns: 1fr; }}
    .results-grid {{ grid-template-columns: 1fr; }}
    .orbitals-grid {{ grid-template-columns: 1fr; }}
    .section {{ padding: 20px; }}
    h1 {{ font-size: 26px; }}
}}
</style>
</head>
<body>
<div class="container">

<h1>{mol_name} - 氧化还原电位</h1>
<p class="subtitle">工作目录: {workdir.name}</p>

<div class="section">
    <h2>分子结构</h2>
    <div class="structures">
        <div class="structure-card">
            <div class="card-header">中性态 (GN)</div>
            <div id="viewer_GN" class="mol-viewer"></div>
            <div class="style-selector">
                <label><input type="radio" name="style_GN" value="stickball" onchange="changeStyle('GN', this.value)" checked> 球棍</label>
                <label><input type="radio" name="style_GN" value="stick" onchange="changeStyle('GN', this.value)"> 棍状</label>
                <label><input type="radio" name="style_GN" value="line" onchange="changeStyle('GN', this.value)"> 线状</label>
            </div>
        </div>
        <div class="structure-card">
            <div class="card-header">氧化态 (OX)</div>
            <div id="viewer_OX" class="mol-viewer"></div>
            <div class="style-selector">
                <label><input type="radio" name="style_OX" value="stickball" onchange="changeStyle('OX', this.value)" checked> 球棍</label>
                <label><input type="radio" name="style_OX" value="stick" onchange="changeStyle('OX', this.value)"> 棍状</label>
                <label><input type="radio" name="style_OX" value="line" onchange="changeStyle('OX', this.value)"> 线状</label>
            </div>
        </div>
        <div class="structure-card">
            <div class="card-header">还原态 (RD)</div>
            <div id="viewer_RD" class="mol-viewer"></div>
            <div class="style-selector">
                <label><input type="radio" name="style_RD" value="stickball" onchange="changeStyle('RD', this.value)" checked> 球棍</label>
                <label><input type="radio" name="style_RD" value="stick" onchange="changeStyle('RD', this.value)"> 棍状</label>
                <label><input type="radio" name="style_RD" value="line" onchange="changeStyle('RD', this.value)"> 线状</label>
            </div>
        </div>
    </div>
</div>

<div class="section">
    <h2>计算结果 (参比常数: 1.24 V)</h2>
    <div class="results-grid">
        <div class="result-card">
            <div class="label">氧化电位</div>
            <div class="value">{e_ox_str}</div>
        </div>
        <div class="result-card">
            <div class="label">还原电位</div>
            <div class="value">{e_red_str}</div>
        </div>
    </div>
</div>

<div class="section">
    <h2>电子结构 (GN/01)</h2>
    <div class="orbitals-grid">
        <div class="orbital-card">
            <div class="label">HOMO</div>
            <div class="value">{homo_str}</div>
        </div>
        <div class="orbital-card">
            <div class="label">LUMO</div>
            <div class="value">{lumo_str}</div>
        </div>
        <div class="orbital-card">
            <div class="label">Gap (LUMO - HOMO)</div>
            <div class="value">{gap_str}</div>
        </div>
    </div>
</div>

<div class="section">
    <h2>各状态能量细分 (Eh)</h2>
    <table>
        <tr>
            <th>状态</th>
            <th>01 G_corr (Eh)</th>
            <th>02 E_high (Eh)</th>
            <th>03 E_gas (Eh)</th>
            <th>04 E_solv (Eh)</th>
            <th>G_total (Eh)</th>
        </tr>
        <tr>
            <td><b>中性态 (GN)</b></td>
            <td>{summary['states'].get('GN', {{}}).get('G_corr', 'N/A')}</td>
            <td>{summary['states'].get('GN', {{}}).get('E_02', 'N/A')}</td>
            <td>{summary['states'].get('GN', {{}}).get('E_03', 'N/A')}</td>
            <td>{summary['states'].get('GN', {{}}).get('E_04', 'N/A')}</td>
            <td><b>{summary['states'].get('GN', {{}}).get('G_total_Eh', 'N/A')}</b></td>
        </tr>
        <tr>
            <td><b>氧化态 (OX)</b></td>
            <td>{summary['states'].get('OX', {{}}).get('G_corr', 'N/A')}</td>
            <td>{summary['states'].get('OX', {{}}).get('E_02', 'N/A')}</td>
            <td>{summary['states'].get('OX', {{}}).get('E_03', 'N/A')}</td>
            <td>{summary['states'].get('OX', {{}}).get('E_04', 'N/A')}</td>
            <td><b>{summary['states'].get('OX', {{}}).get('G_total_Eh', 'N/A')}</b></td>
        </tr>
        <tr>
            <td><b>还原态 (RD)</b></td>
            <td>{summary['states'].get('RD', {{}}).get('G_corr', 'N/A')}</td>
            <td>{summary['states'].get('RD', {{}}).get('E_02', 'N/A')}</td>
            <td>{summary['states'].get('RD', {{}}).get('E_03', 'N/A')}</td>
            <td>{summary['states'].get('RD', {{}}).get('E_04', 'N/A')}</td>
            <td><b>{summary['states'].get('RD', {{}}).get('G_total_Eh', 'N/A')}</b></td>
        </tr>
    </table>
</div>

<div class="section">
    <h2>计算方法</h2>
    <div class="flowchart-container">
        <svg viewBox="0 0 720 560" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Noto Sans SC', 'Inter', sans-serif;">
            <defs>
                <marker id="flowArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
                    <path d="M1 1 L7 4 L1 7 Z" fill="#8a9ab5" stroke="none"/>
                </marker>
            </defs>
            <g>
                <rect x="40" y="20" width="640" height="88" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
                <circle cx="72" cy="64" r="20" fill="#1a3a6b"/>
                <text x="72" y="69" text-anchor="middle" fill="#fff" font-size="13" font-weight="700" font-family="Inter, sans-serif">01</text>
                <text x="108" y="54" fill="#1c2a3a" font-size="15" font-weight="600">结构优化 + 频率计算 + 热力学校正量</text>
                <text x="108" y="80" fill="#5a6c80" font-size="13" font-family="Inter, 'Courier New', monospace">B3LYP/6-311+G(d) + D3BJ + CPCM(eps=18.5)</text>
                <rect x="555" y="50" width="85" height="30" rx="15" fill="#eef3fb" stroke="#3a6ab5" stroke-width="1"/>
                <text x="597" y="69" text-anchor="middle" fill="#1a3a6b" font-size="13" font-weight="600" font-family="Inter, monospace">G_corr</text>
            </g>
            <line x1="360" y1="108" x2="360" y2="148" stroke="#8a9ab5" stroke-width="2" stroke-linecap="round" marker-end="url(#flowArrow)" fill="none"/>
            <g>
                <rect x="40" y="150" width="640" height="88" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
                <circle cx="72" cy="194" r="20" fill="#1a3a6b"/>
                <text x="72" y="199" text-anchor="middle" fill="#fff" font-size="13" font-weight="700" font-family="Inter, sans-serif">02</text>
                <text x="108" y="184" fill="#1c2a3a" font-size="15" font-weight="600">高精度气相单点能</text>
                <text x="108" y="210" fill="#5a6c80" font-size="13" font-family="Inter, 'Courier New', monospace">RI-B2PLYP/ma-def2-TZVP + D3BJ</text>
                <rect x="568" y="180" width="72" height="30" rx="15" fill="#eef3fb" stroke="#3a6ab5" stroke-width="1"/>
                <text x="604" y="199" text-anchor="middle" fill="#1a3a6b" font-size="13" font-weight="600" font-family="Inter, monospace">E_02</text>
            </g>
            <line x1="360" y1="238" x2="360" y2="278" stroke="#8a9ab5" stroke-width="2" stroke-linecap="round" marker-end="url(#flowArrow)" fill="none"/>
            <g>
                <rect x="40" y="280" width="640" height="88" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
                <circle cx="72" cy="324" r="20" fill="#1a3a6b"/>
                <text x="72" y="329" text-anchor="middle" fill="#fff" font-size="13" font-weight="700" font-family="Inter, sans-serif">03</text>
                <text x="108" y="314" fill="#1c2a3a" font-size="15" font-weight="600">气相基准单点能</text>
                <text x="108" y="340" fill="#5a6c80" font-size="13" font-family="Inter, 'Courier New', monospace">M062X/6-31G(d) + D3Zero</text>
                <rect x="568" y="310" width="72" height="30" rx="15" fill="#eef3fb" stroke="#3a6ab5" stroke-width="1"/>
                <text x="604" y="329" text-anchor="middle" fill="#1a3a6b" font-size="13" font-weight="600" font-family="Inter, monospace">E_03</text>
            </g>
            <line x1="360" y1="368" x2="360" y2="408" stroke="#8a9ab5" stroke-width="2" stroke-linecap="round" marker-end="url(#flowArrow)" fill="none"/>
            <g>
                <rect x="40" y="410" width="640" height="88" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
                <circle cx="72" cy="454" r="20" fill="#1a3a6b"/>
                <text x="72" y="459" text-anchor="middle" fill="#fff" font-size="13" font-weight="700" font-family="Inter, sans-serif">04</text>
                <text x="108" y="444" fill="#1c2a3a" font-size="15" font-weight="600">溶剂中基准单点能</text>
                <text x="108" y="470" fill="#5a6c80" font-size="13" font-family="Inter, 'Courier New', monospace">M062X/6-31G(d) + D3Zero + CPCM(eps=18.5)</text>
                <rect x="568" y="440" width="72" height="30" rx="15" fill="#eef3fb" stroke="#3a6ab5" stroke-width="1"/>
                <text x="604" y="459" text-anchor="middle" fill="#1a3a6b" font-size="13" font-weight="600" font-family="Inter, monospace">E_04</text>
            </g>
        </svg>
    </div>
    <div class="method-formula">
        <div class="formula-box">
            <p><span class="label">溶剂化自由能:</span> <span class="value">G<sub>solv</sub> = G<sub>corr</sub>(01) + E<sub>02</sub> + (E<sub>04</sub> — E<sub>03</sub>)</span></p>
            <p><span class="label">氧化电位:</span> <span class="value">E<sub>ox</sub> = (G<sub>OX</sub> — G<sub>GN</sub>) × 27.2114 — 1.240 V</span></p>
            <p><span class="label">还原电位:</span> <span class="value">E<sub>red</sub> = —(G<sub>RD</sub> — G<sub>GN</sub>) × 27.2114 — 1.240 V</span></p>
        </div>
    </div>
</div>

<p class="footer">Powered by 3Dmol.js · Generated by ORCA Redox Pipeline</p>

</div>

<script>
const xyzData = {js_xyz_data_str};

viewers = {{}};
taskTypes = ["GN", "OX", "RD"];
taskTypes.forEach(function(s) {{
    if (!xyzData[s]) return;
    var element = $("#viewer_" + s);
    if (element.length === 0) return;
    var config = {{ backgroundColor: "#fafbfc" }};
    var viewer = $3Dmol.createViewer(element, config);
    viewer.addModel(xyzData[s], "xyz");
    viewer.setStyle({{ stick: {{ radius: 0.15, colorscheme: "Jmol" }}, sphere: {{ scale: 0.25, colorscheme: "Jmol" }} }});
    viewer.zoomTo();
    viewer.render();
    viewers[s] = viewer;
}});

function changeStyle(state, style) {{
    var viewer = viewers[state];
    if (!viewer) return;
    viewer.removeAllSurfaces();
    if (style === "stickball") {{
        viewer.setStyle({{ stick: {{ radius: 0.15, colorscheme: "Jmol" }}, sphere: {{ scale: 0.25, colorscheme: "Jmol" }} }});
    }} else if (style === "stick") {{
        viewer.setStyle({{ stick: {{ radius: 0.2, colorscheme: "Jmol" }} }});
    }} else if (style === "line") {{
        viewer.setStyle({{ line: {{}} }});
    }}
    viewer.render();
}}
</script>
</body>
</html>"""

    out_path = workdir / output_filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return out_path

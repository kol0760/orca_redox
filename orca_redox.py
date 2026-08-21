#!/usr/bin/env python3
"""
ORCA Redox Potential Calculation Workflow
------------------------------------------
Automated workflow for computing oxidation/reduction potentials using ORCA
with a 4-job Slurm dependency architecture (or local execution).

Features:
- Isolated retry folders for robust self-healing (SCF, MaxIter, Imaginary Frequencies)
- Non-blocking GN fallback: If GN fails, OX and RD automatically fall back to initial geometry
- Independent calculation & partial analysis: Even if OX or RD fails, the surviving potential is computed and reported
"""

import os
import sys
import argparse
import subprocess
import re
import yaml
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

# Pandas import for CSV report
try:
    import pandas as pd
except ImportError:
    pd = None

HARTREE_TO_EV = 27.211386245988
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.exists():
        return {
            "resources": {"cores": 32, "memory_per_core_mb": 2000, "slurm": {"partition": "", "time": "24:00:00", "analysis_time": "00:30:00"}},
            "solvation": {"epsilon": 18.5, "refrac": 1.378},
            "methods": {
                "step01": {"simple_input": "! B3LYP 6-311+G(d) D3BJ OPT FREQ CPCM def2/J RIJCOSX tightSCF noautostart miniprint", "scf_maxiter": 512, "geom_maxiter": 512},
                "step02": {"simple_input": "! RI-B2PLYP D3BJ ma-def2-TZVP AutoAux TightSCF", "scf_maxiter": 512},
                "step03": {"simple_input": "! M062X D3Zero 6-31G(d) TightSCF DefGrid3", "scf_maxiter": 512},
                "step04": {"simple_input": "! M062X D3Zero 6-31G(d) TightSCF DefGrid3 CPCM", "scf_maxiter": 512}
            },
            "reference_electrodes": {"SHE": 4.281, "Li_Li_plus": 1.24, "Fc_Fc_plus": 4.681},
            "auto_fix": {"max_attempts": 3}
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_molecule_complexity(input_path_or_smiles: str) -> Dict[str, Any]:
    """Analyze molecular size and flexibility to determine if gas pre-opt is needed."""
    info = {"num_heavy_atoms": 0, "num_rotatable_bonds": 0, "need_preopt": False}
    if Chem is None:
        return info
        
    mol = None
    if os.path.isfile(input_path_or_smiles) or input_path_or_smiles.endswith(".xyz"):
        # Count atoms from xyz file
        try:
            with open(input_path_or_smiles, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            if len(lines) >= 3:
                heavy_count = sum(1 for l in lines[2:] if l.strip() and not l.strip().split()[0].upper().startswith("H"))
                info["num_heavy_atoms"] = heavy_count
                if heavy_count >= 15:
                    info["need_preopt"] = True
        except Exception:
            pass
        return info
    else:
        # SMILES
        try:
            mol = Chem.MolFromSmiles(input_path_or_smiles)
            if mol:
                from rdkit.Chem import Lipinski
                info["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
                info["num_rotatable_bonds"] = Lipinski.NumRotatableBonds(mol)
                if info["num_heavy_atoms"] >= 15 or info["num_rotatable_bonds"] >= 4:
                    info["need_preopt"] = True
        except Exception:
            pass
    return info


def smiles_to_xyz(smiles: str, output_xyz: Path, name: str = "mol") -> bool:
    """Generate 3D conformer from SMILES using RDKit."""
    if Chem is None or AllChem is None:
        raise RuntimeError("RDKit is required to convert SMILES to 3D XYZ. Please install rdkit.")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res < 0:
        res = AllChem.EmbedMolecule(mol)
        if res < 0:
            raise RuntimeError(f"Failed to generate 3D conformer for SMILES: {smiles}")
    
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    
    conf = mol.GetConformer()
    with open(output_xyz, "w", encoding="utf-8") as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"{name} generated from SMILES {smiles}\n")
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol():<3} {pos.x:14.8f} {pos.y:14.8f} {pos.z:14.8f}\n")
    return True


def write_inp_file(
    filepath: Path,
    step_num: str,
    charge: int,
    mult: int,
    xyz_filename: str,
    cfg: Dict[str, Any],
    cores: int
):
    """Write ORCA .inp file for step 01, 02, 03, or 04, or preopt."""
    if step_num == "preopt":
        lines = [
            "! B3LYP 6-31G(d) D3BJ LooseOpt def2/J RIJCOSX tightSCF noautostart miniprint",
            "",
            f"%PAL nprocs {cores} END",
            "",
            "%SCF MaxIter 512 END",
            "%GEOM MaxIter 512 END",
            "",
            f"* xyzfile {charge} {mult} {xyz_filename}",
            ""
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    step_key = f"step{step_num}"
    step_cfg = cfg["methods"][step_key]
    simple_inp = step_cfg["simple_input"]
    scf_maxiter = step_cfg.get("scf_maxiter", 512)
    geom_maxiter = step_cfg.get("geom_maxiter", 512)
    
    lines = []
    lines.append(f"{simple_inp}")
    lines.append("")
    lines.append(f"%PAL nprocs {cores} END")
    
    if "CPCM" in simple_inp.upper():
        eps = cfg["solvation"]["epsilon"]
        refrac = cfg["solvation"]["refrac"]
        lines.append("")
        lines.append("%CPCM")
        lines.append(f"  epsilon {eps}")
        lines.append(f"  refrac {refrac}")
        lines.append("END")
    
    lines.append("")
    lines.append(f"%SCF MaxIter {scf_maxiter} END")
    if "OPT" in simple_inp.upper():
        lines.append(f"%GEOM MaxIter {geom_maxiter} END")
    
    lines.append("")
    lines.append(f"* xyzfile {charge} {mult} {xyz_filename}")
    lines.append("")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_slurm_script(
    workdir: Path,
    state: str,
    cores: int,
    cfg: Dict[str, Any],
    skip01: bool = False,
    is_analysis: bool = False,
    mol_name: str = ""
) -> Path:
    """Generate Slurm job script strictly matching the user's template."""
    slurm_cfg = cfg["resources"].get("slurm", {})
    partition = slurm_cfg.get("partition", "hpc")
    time_limit = slurm_cfg.get("analysis_time", "24:00:00") if is_analysis else slurm_cfg.get("time", "24:00:00")
    env_source = slurm_cfg.get("env_source", "source /fastone/softwares/hpc-kits/hpc-kits.sh")
    orca_path = slurm_cfg.get("orca_path", "/fastone/softwares/Orca/orca-6.1.0-f.0_linux_x86-64/bin/orca")
    nodes = cfg["resources"].get("nodes", 1)
    script_path = workdir / f"run_{state.lower()}.slurm"
    smart_opt_path = Path(__file__).parent / "smart_optimizer.py"
    
    job_name = f"{mol_name}_{state}" if mol_name else f"ORCA_{state}"
    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH -J {job_name}")
    if is_analysis:
        lines.append("#SBATCH --ntasks-per-node=1")
    else:
        lines.append(f"#SBATCH --ntasks-per-node={cores}")
    lines.append(f"#SBATCH --nodes={nodes}")
    if partition:
        lines.append(f"#SBATCH --partition {partition}")
    if time_limit:
        lines.append(f"#SBATCH --time={time_limit}")
    lines.append("")
    lines.append("")
    lines.append("# 应用command的绝对路径")
    lines.append("")
    lines.append("set -x")
    lines.append("")
    if env_source:
        lines.append(f"{env_source}")
    lines.append(f'orca_path="{orca_path}"')
    lines.append("")
    
    if is_analysis:
        lines.append('cd "${SLURM_SUBMIT_DIR}"')
        lines.append("")
        lines.append(f"python3 {Path(__file__).resolve()} -n {mol_name} --analysis")
        lines.append("exit 0")
    elif state == "PREOPT":
        lines.append(f'cd "${{SLURM_SUBMIT_DIR}}/PREOPT"')
        lines.append("")
        lines.append("# --- 气相粗糙预优化 (LooseOpt B3LYP/6-31G*) ---")
        lines.append('echo "[*] Running gas-phase pre-optimization in PREOPT/..."')
        lines.append('$orca_path 01.inp > 01.out 2>&1')
        lines.append("if grep -q '\\*\\*\\*\\*ORCA TERMINATED NORMALLY\\*\\*\\*\\*' 01.out; then")
        lines.append("    echo '[✓] Pre-optimization converged successfully.'")
        lines.append("else")
        lines.append("    echo '[!] Pre-optimization not fully converged. Downstream GN will fallback to trajectory/initial geometry.'")
        lines.append("fi")
        lines.append("exit 0")
    else:
        state_dir = workdir / state
        lines.append(f'cd "${{SLURM_SUBMIT_DIR}}/{state}"')
        lines.append("")
        
        # Geometry Inheritance logic for GN, OX, RD
        if state == "GN":
            lines.append("# --- GN 结构继承 (优先继承 PREOPT/01.xyz, 失败则回退) ---")
            lines.append("if [ ! -f mol.xyz ]; then")
            lines.append("    if [ -f ../PREOPT/01.xyz ] && grep -q '\\*\\*\\*\\*ORCA TERMINATED NORMALLY\\*\\*\\*\\*' ../PREOPT/01.out 2>/dev/null; then")
            lines.append("        echo '[i] Inheriting pre-optimized geometry from PREOPT/01.xyz'")
            lines.append("        cp ../PREOPT/01.xyz ./mol.xyz")
            lines.append("    elif [ -f ../PREOPT/01_trj.xyz ]; then")
            lines.append("        echo '[!] PREOPT did not terminate normally. Extracting last frame from PREOPT/01_trj.xyz'")
            lines.append("        python3 -c 'from smart_optimizer import extract_last_geometry_from_trj, write_xyz_file; from pathlib import Path; atoms=extract_last_geometry_from_trj(Path(\"../PREOPT/01_trj.xyz\")); write_xyz_file(Path(\"mol.xyz\"), atoms) if atoms else None'")
            lines.append("    fi")
            lines.append(f"    if [ ! -f mol.xyz ] && [ -f ../{mol_name}.xyz ]; then")
            lines.append(f"        echo '[i] Falling back to initial starting geometry'")
            lines.append(f"        cp ../{mol_name}.xyz ./mol.xyz")
            lines.append("    fi")
            lines.append("fi")
            lines.append("")
        elif state in ["OX", "RD"]:
            lines.append("# --- 结构继承与回退策略 ---")
            lines.append("if [ ! -f mol.xyz ]; then")
            lines.append("    if [ -f ../GN/01.xyz ] && grep -q '\\*\\*\\*\\*ORCA TERMINATED NORMALLY\\*\\*\\*\\*' ../GN/01.out 2>/dev/null; then")
            lines.append("        echo '[i] Inheriting optimized geometry from GN/01.xyz'")
            lines.append("        cp ../GN/01.xyz ./mol.xyz")
            lines.append("    else")
            lines.append("        echo '[!] GN/01.xyz not converged or missing. Falling back to initial starting geometry'")
            lines.append(f"        if [ -f ../GN/{mol_name}.xyz ]; then")
            lines.append(f"            cp ../GN/{mol_name}.xyz ./mol.xyz")
            lines.append(f"        elif [ -f ../{mol_name}.xyz ]; then")
            lines.append(f"            cp ../{mol_name}.xyz ./mol.xyz")
            lines.append("        fi")
            lines.append("    fi")
            lines.append("fi")
            lines.append("")
        
        if not skip01:
            lines.append("# Step 01: Opt + Freq (智能自救优化)")
            lines.append(f'python3 {smart_opt_path.resolve()} --step 01 --cores {cores} --orca_cmd "$orca_path"')
            lines.append("if [ $? -ne 0 ]; then")
            lines.append(f"    echo 'Step 01 optimization failed in {state}!' >&2")
            lines.append("    exit 1")
            lines.append("fi")
            lines.append("")
        
        lines.append("# Steps 02, 03, 04 (单点能计算)")
        lines.append("for step in 02 03 04; do")
        lines.append(f'    python3 {smart_opt_path.resolve()} --step ${{step}} --cores {cores} --orca_cmd "$orca_path"')
        lines.append('    if [ $? -ne 0 ]; then')
        lines.append('        echo "Step ${step} failed in $PWD!" >&2')
        lines.append('        exit 1')
        lines.append('    fi')
        lines.append("done")
        lines.append("")
        lines.append(f"echo '{state} calculations completed successfully.'")
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    os.chmod(script_path, 0o755)
    return script_path


def parse_orca_output(state_dir: Path) -> Dict[str, Any]:
    """Parse output files 01.out ~ 04.out for a state directory."""
    results = {
        "converged": True,
        "imaginary_freqs": [],
        "G_corr": None,      # Thermal correction to Gibbs free energy (Hartree)
        "ZPE": None,         # Zero point energy (Hartree)
        "E_01": None,        # Step 01 final single point energy
        "E_02": None,        # Step 02 high-level gas energy
        "E_03": None,        # Step 03 low-level gas energy
        "E_04": None,        # Step 04 low-level solv energy
        "errors": []
    }
    
    # 1. Parse 01.out (Opt + Freq)
    f01 = state_dir / "01.out"
    if not f01.exists():
        results["converged"] = False
        results["errors"].append("01.out does not exist")
    else:
        with open(f01, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        if "****ORCA TERMINATED NORMALLY****" not in content:
            results["converged"] = False
            results["errors"].append("01.out did not terminate normally")
        
        # Check imaginary frequencies
        freq_matches = re.findall(r":\s+([-+]?\d+\.\d+)\s+cm\*\*-1", content)
        if freq_matches:
            for freq_str in freq_matches:
                val = float(freq_str)
                if val < -50.0:  # Threshold for serious imaginary frequency
                    results["imaginary_freqs"].append(val)
        
        g_corr_m = re.search(r"Thermal correction to Gibbs free energy\s+\.\.\.\s+([-+]?\d+\.\d+)\s+Eh", content)
        if g_corr_m:
            results["G_corr"] = float(g_corr_m.group(1))
        else:
            g_corr_m2 = re.search(r"G-E\(el\)\s+\.\.\.\s+([-+]?\d+\.\d+)\s+Eh", content)
            if g_corr_m2:
                results["G_corr"] = float(g_corr_m2.group(1))
        
        e_m = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", content)
        if e_m:
            results["E_01"] = float(e_m[-1])

    # 2. Parse 02.out, 03.out, 04.out
    for step, key in [("02", "E_02"), ("03", "E_03"), ("04", "E_04")]:
        fpath = state_dir / f"{step}.out"
        if not fpath.exists():
            results["converged"] = False
            results["errors"].append(f"{step}.out does not exist")
            continue
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            c = f.read()
        if "****ORCA TERMINATED NORMALLY****" not in c:
            results["converged"] = False
            results["errors"].append(f"{step}.out did not terminate normally")
        
        e_m = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", c)
        if e_m:
            results[key] = float(e_m[-1])
        else:
            results["converged"] = False
            results["errors"].append(f"Could not parse FINAL SINGLE POINT ENERGY from {step}.out")
            
    return results


def parse_orbitals_from_orca(output_text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse HOMO, LUMO, and Gap (eV) from ORCA output (supports both RKS and UKS)."""
    # Look for last ORBITAL ENERGIES block
    orb_blocks = re.findall(r"ORBITAL ENERGIES\s*[-=]+(.*?)(?:--------|\*\*\*\*ORCA|TOTAL RUN TIME)", output_text, re.DOTALL)
    if not orb_blocks:
        return None, None, None
        
    block = orb_blocks[-1]
    
    # Check if UKS (SPIN UP / SPIN DOWN)
    if "SPIN UP ORBITALS" in block or "SPIN DOWN ORBITALS" in block:
        # Separate spin channels
        parts = re.split(r"SPIN (?:UP|DOWN) ORBITALS", block)
        homo_candidates = []
        lumo_candidates = []
        for p in parts:
            p_homo, p_lumo = None, None
            for line in p.strip().split("\n"):
                tokens = line.strip().split()
                if len(tokens) >= 4:
                    try:
                        occ = float(tokens[1])
                        e_ev = float(tokens[3])
                        if occ > 0.1: # Occupied in spin channel
                            p_homo = e_ev
                        elif occ <= 0.1 and p_lumo is None:
                            p_lumo = e_ev
                    except ValueError:
                        continue
            if p_homo is not None:
                homo_candidates.append(p_homo)
            if p_lumo is not None:
                lumo_candidates.append(p_lumo)
        
        last_occ_ev = max(homo_candidates) if homo_candidates else None
        first_virt_ev = min(lumo_candidates) if lumo_candidates else None
    else:
        # Standard RKS
        lines = block.strip().split("\n")
        last_occ_ev = None
        first_virt_ev = None
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) >= 4:
                try:
                    occ = float(tokens[1])
                    e_ev = float(tokens[3])
                    if occ > 0.5:
                        last_occ_ev = e_ev
                    elif occ <= 0.5 and first_virt_ev is None:
                        first_virt_ev = e_ev
                except ValueError:
                    continue
                    
    gap = (first_virt_ev - last_occ_ev) if (first_virt_ev is not None and last_occ_ev is not None) else None
    return last_occ_ev, first_virt_ev, gap


def calculate_redox_report(
    workdir: Path,
    cfg: Dict[str, Any],
    only_ox: bool = False,
    only_rd: bool = False
) -> Dict[str, Any]:
    """Calculate Gibbs free energy and Redox potentials with maximum fault tolerance."""
    states_to_check = ["GN"]
    if not only_rd:
        states_to_check.append("OX")
    if not only_ox:
        states_to_check.append("RD")
        
    state_results = {}
    for st in states_to_check:
        st_dir = workdir / st
        if not st_dir.exists():
            state_results[st] = {"converged": False, "errors": ["Directory missing"]}
            continue
        state_results[st] = parse_orca_output(st_dir)
    
    # Calculate G_solv for each converged state
    # Formula: G_solv = E_high_gas(02) + G_corr(01) + [E_low_solv(04) - E_low_gas(03)]
    for st, data in state_results.items():
        if data["converged"] and data["E_02"] is not None and data["G_corr"] is not None and data["E_04"] is not None and data["E_03"] is not None:
            delta_G_solv_correction = data["E_04"] - data["E_03"]
            G_total = data["E_02"] + data["G_corr"] + delta_G_solv_correction
            data["delta_G_solv_corr"] = delta_G_solv_correction
            data["G_total_Eh"] = G_total
        else:
            data["G_total_Eh"] = None
    
    # Parse HOMO / LUMO strictly from GN/01.out (fallback to None / N/A if missing)
    homo_ev, lumo_ev, gap_ev = None, None, None
    gn_01_out = workdir / "GN" / "01.out"
    if gn_01_out.exists():
        with open(gn_01_out, "r", encoding="utf-8", errors="ignore") as f:
            c = f.read()
        homo_ev, lumo_ev, gap_ev = parse_orbitals_from_orca(c)
    
    # Compute Potentials (Fixed 1.24 V reference)
    v_ref_she = cfg["reference_electrodes"].get("SHE", 4.281)
    v_ref_li = cfg["reference_electrodes"].get("Li_Li_plus", 1.24)
    v_ref_fc = cfg["reference_electrodes"].get("Fc_Fc_plus", 4.681)
    
    summary = {
        "molecule": workdir.name,
        "states": state_results,
        "HOMO_eV": homo_ev,
        "LUMO_eV": lumo_ev,
        "Gap_eV": gap_ev,
        "E_ox_SHE_V": None,
        "E_ox_Li_V": None,
        "E_ox_Fc_V": None,
        "E_red_SHE_V": None,
        "E_red_Li_V": None,
        "E_red_Fc_V": None,
        "Delta_G_ox_eV": None,
        "Delta_G_red_eV": None,
    }
    
    gn_g = state_results.get("GN", {}).get("G_total_Eh")
    
    # Oxidation Potential: Delta_G_ox = G(OX) - G(GN)
    if "OX" in state_results and state_results["OX"].get("G_total_Eh") is not None:
        if gn_g is not None:
            delta_g_ox_eh = state_results["OX"]["G_total_Eh"] - gn_g
            delta_g_ox_ev = delta_g_ox_eh * HARTREE_TO_EV
            summary["Delta_G_ox_eV"] = delta_g_ox_ev
            summary["E_ox_SHE_V"] = delta_g_ox_ev - v_ref_she
            summary["E_ox_Li_V"] = delta_g_ox_ev - v_ref_li
            summary["E_ox_Fc_V"] = delta_g_ox_ev - v_ref_fc

    # Reduction Potential: Delta_G_red = G(GN) - G(RD)
    if "RD" in state_results and state_results["RD"].get("G_total_Eh") is not None:
        if gn_g is not None:
            delta_g_red_eh = gn_g - state_results["RD"]["G_total_Eh"]
            delta_g_red_ev = delta_g_red_eh * HARTREE_TO_EV
            summary["Delta_G_red_eV"] = delta_g_red_ev
            summary["E_red_SHE_V"] = -delta_g_red_ev - v_ref_she
            summary["E_red_Li_V"] = -delta_g_red_ev - v_ref_li
            summary["E_red_Fc_V"] = -delta_g_red_ev - v_ref_fc
        
    return summary


def display_and_save_summary(summary: Dict[str, Any], workdir: Path):
    """Print formatted terminal table and save JSON, CSV, and HTML reports."""
    mol = summary["molecule"]
    print("\n" + "=" * 70)
    print(f"         ORCA REDOX POTENTIAL SUMMARY: {mol}")
    print("=" * 70)
    
    for st, res in summary["states"].items():
        status = "CONVERGED" if res["converged"] else f"FAILED ({', '.join(res['errors'])})"
        g_val = f"{res['G_total_Eh']:.6f} Eh" if res.get('G_total_Eh') is not None else "N/A"
        im_freq = f"None" if not res.get("imaginary_freqs") else f"{len(res['imaginary_freqs'])} freqs (< -50 cm^-1: {res['imaginary_freqs']})"
        print(f"  • State [{st}]: Status={status}")
        print(f"      G_total   : {g_val}")
        print(f"      Imag Freqs: {im_freq}")
    
    print("-" * 70)
    # Electronic structure
    homo = f"{summary['HOMO_eV']:.4f} eV" if summary.get("HOMO_eV") is not None else "N/A"
    lumo = f"{summary['LUMO_eV']:.4f} eV" if summary.get("LUMO_eV") is not None else "N/A"
    gap = f"{summary['Gap_eV']:.4f} eV" if summary.get("Gap_eV") is not None else "N/A"
    print(f"  [Frontier Orbitals (GN)]  HOMO: {homo}  |  LUMO: {lumo}  |  Gap: {gap}")
    print("-" * 70)
    
    # Potential results (Fixed 1.24 V reference)
    if summary["E_ox_Li_V"] is not None:
        print(f"  [Oxidation Potential]  E_ox  = {summary['E_ox_Li_V']:8.4f} V (ref: 1.240 V)")
    else:
        print("  [Oxidation Potential]  E_ox  : Not available (GN or OX state incomplete)")
        
    if summary["E_red_Li_V"] is not None:
        print(f"  [Reduction Potential]  E_red = {summary['E_red_Li_V']:8.4f} V (ref: 1.240 V)")
    else:
        print("  [Reduction Potential]  E_red : Not available (GN or RD state incomplete)")
    print("=" * 70 + "\n")
    
    # Save JSON
    json_path = workdir / "redox_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON report to: {json_path}")
    
    # Save CSV
    if pd is not None:
        row = {
            "Molecule": mol,
            "E_ox_vs_Li (V)": summary["E_ox_Li_V"],
            "E_red_vs_Li (V)": summary["E_red_Li_V"],
            "HOMO (eV)": summary.get("HOMO_eV"),
            "LUMO (eV)": summary.get("LUMO_eV"),
            "Gap (eV)": summary.get("Gap_eV"),
            "GN_G_total (Eh)": summary["states"].get("GN", {}).get("G_total_Eh"),
            "OX_G_total (Eh)": summary["states"].get("OX", {}).get("G_total_Eh"),
            "RD_G_total (Eh)": summary["states"].get("RD", {}).get("G_total_Eh"),
        }
        df = pd.DataFrame([row])
        csv_path = workdir / "redox_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV report to: {csv_path}")

    # Generate Specific Interactive HTML Report (e.g. FEC_addLi_redox_report.html and report.html)
    try:
        from html_reporter import generate_html_report
        specific_html_name = f"{mol}_redox_report.html"
        html_path = generate_html_report(summary, workdir, output_filename=specific_html_name)
        # Also copy/generate report.html for convenience
        generate_html_report(summary, workdir, output_filename="report.html")
        print(f"Saved Interactive HTML report to: {html_path}")
    except Exception as e:
        print(f"[Warning] Failed to generate HTML report: {e}")


def submit_slurm_jobs(workdir: Path, mol_name: str, only_ox: bool = False, only_rd: bool = False) -> Dict[str, str]:
    """
    Submit Slurm jobs in sequence:
    Optional: PREOPT -> GN (dep afterany:PREOPT) -> [OX, RD] (dep afterany:GN) -> Analysis (dep afterany:all)
    """
    job_ids = {}
    
    # 0. Submit PREOPT if present
    job_preopt = None
    if (workdir / "run_preopt.slurm").exists():
        res_pre = subprocess.run(["sbatch", "run_preopt.slurm"], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res_pre.returncode == 0:
            m_pre = re.search(r"Submitted batch job (\d+)", res_pre.stdout)
            if m_pre:
                job_preopt = m_pre.group(1)
                job_ids["PREOPT"] = job_preopt
                print(f"  [Slurm] Submitted Job 0 (PREOPT): Job ID {job_preopt}")
                
    # 1. Submit GN (Neutral state)
    gn_cmd = ["sbatch"]
    if job_preopt:
        gn_cmd.append(f"--dependency=afterany:{job_preopt}")
    gn_cmd.append("run_gn.slurm")
    
    res_gn = subprocess.run(gn_cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res_gn.returncode != 0:
        print(f"[Error] Failed to submit run_gn.slurm: {res_gn.stderr}")
        return job_ids
    
    m_gn = re.search(r"Submitted batch job (\d+)", res_gn.stdout)
    if not m_gn:
        print(f"[Error] Could not parse job ID from: {res_gn.stdout}")
        return job_ids
    job_gn = m_gn.group(1)
    job_ids["GN"] = job_gn
    if job_preopt:
        print(f"  [Slurm] Submitted Job 1 (GN, dep afterany:{job_preopt}): Job ID {job_gn}")
    else:
        print(f"  [Slurm] Submitted Job 1 (GN): Job ID {job_gn}")
    
    dep_for_analysis = [job_gn]
    
    # 2. Submit OX (dependency: afterany:job_gn) -> If GN fails, OX falls back to initial coords
    if not only_rd and (workdir / "run_ox.slurm").exists():
        res_ox = subprocess.run(
            ["sbatch", f"--dependency=afterany:{job_gn}", "run_ox.slurm"],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res_ox.returncode == 0:
            m_ox = re.search(r"Submitted batch job (\d+)", res_ox.stdout)
            if m_ox:
                job_ox = m_ox.group(1)
                job_ids["OX"] = job_ox
                dep_for_analysis.append(job_ox)
                print(f"  [Slurm] Submitted Job 2 (OX, dep afterany:{job_gn}): Job ID {job_ox}")
    
    # 3. Submit RD (dependency: afterany:job_gn) -> If GN fails, RD falls back to initial coords
    if not only_ox and (workdir / "run_rd.slurm").exists():
        res_rd = subprocess.run(
            ["sbatch", f"--dependency=afterany:{job_gn}", "run_rd.slurm"],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res_rd.returncode == 0:
            m_rd = re.search(r"Submitted batch job (\d+)", res_rd.stdout)
            if m_rd:
                job_rd = m_rd.group(1)
                job_ids["RD"] = job_rd
                dep_for_analysis.append(job_rd)
                print(f"  [Slurm] Submitted Job 3 (RD, dep afterany:{job_gn}): Job ID {job_rd}")
                
    # 4. Submit Analysis (dependency: afterany:all_submitted_jobs) -> Guarantees report generation
    if (workdir / "run_analysis.slurm").exists():
        all_jobs = [job_gn] + dep_for_analysis
        dep_str = ":".join(all_jobs)
        res_an = subprocess.run(
            ["sbatch", f"--dependency=afterany:{dep_str}", "run_analysis.slurm"],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res_an.returncode == 0:
            m_an = re.search(r"Submitted batch job (\d+)", res_an.stdout)
            if m_an:
                job_an = m_an.group(1)
                job_ids["Analysis"] = job_an
                print(f"  [Slurm] Submitted Job 4 (Analysis, dep afterany:{dep_str}): Job ID {job_an}")
                
    return job_ids


def main():
    parser = argparse.ArgumentParser(
        description="ORCA Redox Potential Calculation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--input", help="输入: XYZ 文件路径或 SMILES 字符串 (--analysis 模式下不需要)")
    parser.add_argument("-n", "--name", help="工作目录名称 (默认取输入文件 stem 或 InChIKey)")
    parser.add_argument("-c", "--charge", type=int, default=0, help="中性态电荷 (默认 0)")
    parser.add_argument("-u", "--uhf", type=int, default=1, help="中性态自旋多重度 (默认 1)")
    parser.add_argument("--cores", type=int, default=32, help="CPU 核数 (默认 32)")
    parser.add_argument("--ox_xyz", help="OX 状态的自定义起始 XYZ 文件 (跳过 GN 坐标继承)")
    parser.add_argument("--rd_xyz", help="RD 状态的自定义起始 XYZ 文件 (跳过 GN 坐标继承)")
    parser.add_argument("--analysis", action="store_true", help="仅分析已有结果 (跳过 ORCA 计算，需 -n 指定目录)")
    parser.add_argument("--skip01", action="store_true", help="跳过所有状态的 01 计算 (假定已手动完成，直接跑 02/03/04)")
    parser.add_argument("--ox", action="store_true", help="仅计算氧化电位")
    parser.add_argument("--rd", action="store_true", help="仅计算还原电位")
    parser.add_argument("--config", default="config.yaml", help="自定义配置文件路径 (默认 config.yaml)")
    parser.add_argument("--no_submit", action="store_true", help="生成输入文件与 Slurm 脚本但不自动提交")

    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    
    # Analysis only mode
    if args.analysis:
        if args.name:
            candidate = Path(args.name)
            # Case 1: candidate exists and contains GN/OX/RD
            if candidate.exists() and ((candidate / "GN").exists() or (candidate / "OX").exists() or (candidate / "RD").exists()):
                workdir = candidate
            # Case 2: relative path from cwd
            elif (Path.cwd() / args.name).exists() and ((Path.cwd() / args.name / "GN").exists() or (Path.cwd() / args.name / "OX").exists()):
                workdir = Path.cwd() / args.name
            # Case 3: already inside the molecular directory (cwd has GN/OX/RD or cwd.name matches args.name)
            elif (Path.cwd() / "GN").exists() or (Path.cwd() / "OX").exists() or Path.cwd().name == args.name:
                workdir = Path.cwd()
            else:
                workdir = candidate
        else:
            workdir = Path.cwd()

        if not workdir.exists():
            print(f"[Error] Target directory {workdir} does not exist for analysis.")
            sys.exit(1)
        summary = calculate_redox_report(workdir, cfg, only_ox=args.ox, only_rd=args.rd)
        display_and_save_summary(summary, workdir)
        sys.exit(0)
        
    # Input verification
    if not args.input:
        print("[Error] Argument -i / --input is required unless running in --analysis mode.")
        parser.print_help()
        sys.exit(1)
        
    input_str = args.input.strip()
    is_xyz_file = os.path.isfile(input_str) or input_str.endswith(".xyz")
    
    if args.name:
        mol_name = args.name
    elif is_xyz_file:
        mol_name = Path(input_str).stem
    else:
        mol_name = "mol_" + re.sub(r"[^a-zA-Z0-9]", "_", input_str)[:12]
        
    workdir = Path(mol_name)
    workdir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Initializing ORCA redox workflow in: {workdir.resolve()}")
    
    # Generate mol.xyz for GN
    gn_xyz_path = workdir / f"{mol_name}.xyz"
    if is_xyz_file:
        shutil.copy(input_str, gn_xyz_path)
    else:
        print(f"[*] Generating 3D structure from SMILES: {input_str}")
        smiles_to_xyz(input_str, gn_xyz_path, name=mol_name)
    
    # Setup States: GN, OX, RD
    states_to_run = ["GN"]
    if not args.rd:
        states_to_run.append("OX")
    if not args.ox:
        states_to_run.append("RD")
        
    charge_gn = args.charge
    mult_gn = args.uhf
    
    charge_ox = charge_gn + 1
    mult_ox = 2 if mult_gn == 1 else (mult_gn - 1 if mult_gn > 1 else 2)
    
    charge_rd = charge_gn - 1
    mult_rd = 2 if mult_gn == 1 else (mult_gn - 1 if mult_gn > 1 else 2)
    
    state_params = {
        "GN": {"charge": charge_gn, "mult": mult_gn, "start_xyz": gn_xyz_path.name},
        "OX": {"charge": charge_ox, "mult": mult_ox, "start_xyz": "mol.xyz"},
        "RD": {"charge": charge_rd, "mult": mult_rd, "start_xyz": "mol.xyz"}
    }
    
    # Custom initial geometries for OX/RD if specified
    if args.ox_xyz and os.path.exists(args.ox_xyz):
        ox_dir = workdir / "OX"
        ox_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.ox_xyz, ox_dir / "mol.xyz")
        state_params["OX"]["start_xyz"] = "mol.xyz"
        
    if args.rd_xyz and os.path.exists(args.rd_xyz):
        rd_dir = workdir / "RD"
        rd_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.rd_xyz, rd_dir / "mol.xyz")
        state_params["RD"]["start_xyz"] = "mol.xyz"

    # Analyze molecular complexity for automatic gas-phase pre-optimization
    mol_complexity = analyze_molecule_complexity(input_str)
    need_preopt = mol_complexity["need_preopt"]
    if need_preopt:
        print(f"[*] Detected large/flexible molecule (Heavy atoms: {mol_complexity['num_heavy_atoms']}, Rotatable bonds: {mol_complexity['num_rotatable_bonds']}).")
        print(f"[*] Automatically enabled isolated gas-phase pre-optimization in PREOPT/ (LooseOpt B3LYP/6-31G*).")
        
        preopt_dir = workdir / "PREOPT"
        preopt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(gn_xyz_path, preopt_dir / gn_xyz_path.name)
        write_inp_file(preopt_dir / "01.inp", "preopt", charge_gn, mult_gn, gn_xyz_path.name, cfg, args.cores)
        generate_slurm_script(workdir, "PREOPT", args.cores, cfg, mol_name=mol_name)
        state_params["GN"]["start_xyz"] = "mol.xyz"

    # Create directories and input files
    for st in states_to_run:
        st_dir = workdir / st
        st_dir.mkdir(parents=True, exist_ok=True)
        
        p = state_params[st]
        
        if st == "GN" and not need_preopt:
            shutil.copy(gn_xyz_path, st_dir / p["start_xyz"])
            
        write_inp_file(st_dir / "01.inp", "01", p["charge"], p["mult"], p["start_xyz"], cfg, args.cores)
        for s in ["02", "03", "04"]:
            write_inp_file(st_dir / f"{s}.inp", s, p["charge"], p["mult"], "01.xyz", cfg, args.cores)

    print(f"[*] Generated input files (01.inp ~ 04.inp) for states: {', '.join(states_to_run)}")
    
    # Generate Slurm scripts
    for st in states_to_run:
        generate_slurm_script(workdir, st, args.cores, cfg, skip01=args.skip01, mol_name=mol_name)
    generate_slurm_script(workdir, "Analysis", 1, cfg, is_analysis=True, mol_name=mol_name)
    print(f"[*] Generated Slurm scripts in {workdir}/")
    
    sbatch_avail = shutil.which("sbatch") is not None
    if sbatch_avail and not args.no_submit:
        print(f"[*] Submitting jobs via Slurm...")
        job_ids = submit_slurm_jobs(workdir, mol_name, only_ox=args.ox, only_rd=args.rd)
        if job_ids:
            print(f"[✓] All jobs submitted successfully: {job_ids}")
        else:
            print(f"[✗] Job submission failed. Please check the Slurm error messages above.")
    else:
        if args.no_submit:
            print("[i] --no_submit flag specified. Slurm scripts ready for manual submission.")
        else:
            print("[i] 'sbatch' command not found. Slurm scripts generated and ready.")
        print(f"\nTo submit manually on cluster with robust fallback dependencies:")
        print(f"  cd {workdir}")
        print(f"  JOB1=$(sbatch run_gn.slurm | awk '{{print $4}}')")
        print(f"  JOB2=$(sbatch --dependency=afterany:$JOB1 run_ox.slurm | awk '{{print $4}}')")
        print(f"  JOB3=$(sbatch --dependency=afterany:$JOB1 run_rd.slurm | awk '{{print $4}}')")
        print(f"  sbatch --dependency=afterany:$JOB1:$JOB2:$JOB3 run_analysis.slurm")


if __name__ == "__main__":
    main()

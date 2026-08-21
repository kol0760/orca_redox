#!/usr/bin/env python3
"""
ORCA Smart Optimizer & Robust Self-Healing Engine
-------------------------------------------------
Handles common ORCA failures with clean retry directories and tiered repair strategies:
1. Imaginary Frequencies (< -50 cm^-1): Mode displacement perturbation.
2. Geometry Optimization MaxIter: Resume from latest trajectory frame with refined trust radius.
3. SCF Convergence Divergence: SlowConv -> SOSCF -> PModel/Shift/Damping.
4. Clean trace: Each retry is executed in an isolated 'retry_XX_<reason>' directory.
"""

import os
import sys
import argparse
import subprocess
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

IMAG_IGNORE_THRESHOLD = -50.0    # Frequencies >= -50.0 cm^-1 are considered numerical noise, ignored
IMAG_REFINE_THRESHOLD = -110.0   # -110.0 <= freq < -50.0: Moderate imag freq -> Use VeryTightOpt + FinalGrid5 on current geom
MAX_REPAIR_ATTEMPTS = 3


def read_file_safe(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def check_normal_termination(output_text: str) -> bool:
    return "****ORCA TERMINATED NORMALLY****" in output_text


def parse_vibrational_frequencies(output_text: str) -> List[float]:
    """Parse vibrational frequencies from ORCA output."""
    freq_block_m = re.findall(r"VIBRATIONAL FREQUENCIES\s*[-=]+(.*?)(?:NORMAL MODES|THERMOCHEMISTRY|IR SPECTRUM)", output_text, re.DOTALL)
    if not freq_block_m:
        raw_matches = re.findall(r"\d+:\s+([-+]?\d+\.\d+)\s+cm\*\*-1", output_text)
        return [float(x) for x in raw_matches]
    
    block = freq_block_m[-1]
    lines = block.strip().split("\n")
    freqs = []
    for line in lines:
        m = re.search(r"^\s*\d+:\s+([-+]?\d+\.\d+)\s+cm\*\*-1", line)
        if m:
            freqs.append(float(m.group(1)))
    return freqs


def parse_imaginary_normal_mode(output_text: str, mode_index: int = 6) -> Optional[List[Tuple[float, float, float]]]:
    """Extract normal mode displacement vectors for an imaginary mode from ORCA output."""
    mode_section_m = re.search(r"NORMAL MODES\s*[-=]+(.*?)(?:IR SPECTRUM|RAMAN SPECTRUM|THERMOCHEMISTRY|\*\*\*\*ORCA)", output_text, re.DOTALL)
    if not mode_section_m:
        return None
    
    section = mode_section_m.group(1)
    lines = section.split("\n")
    displacements = {}
    current_col_modes = []
    current_atom_idx = None
    
    i = 0
    target_col = -1
    while i < len(lines):
        line = lines[i].strip()
        tokens = line.split()
        if not tokens:
            i += 1
            continue
            
        # Check if header line with mode indices (e.g. 6 7 8 9 10 11)
        if all(t.isdigit() for t in tokens):
            current_col_modes = [int(t) for t in tokens]
            if mode_index in current_col_modes:
                target_col = current_col_modes.index(mode_index)
            else:
                target_col = -1
            current_atom_idx = None
            i += 1
            continue
            
        if target_col != -1:
            # Case 1: First line of an atom, e.g. ["0", "C", "x", "0.123", "0.000", ...]
            if tokens[0].isdigit() and len(tokens) >= (len(current_col_modes) + 3):
                try:
                    current_atom_idx = int(tokens[0])
                    if current_atom_idx not in displacements:
                        displacements[current_atom_idx] = []
                    val = float(tokens[3 + target_col])
                    displacements[current_atom_idx].append(val)
                except (ValueError, IndexError):
                    pass
            # Case 2: y or z line, e.g. ["y", "-0.654", "0.000", ...] or ["z", ...]
            elif tokens[0] in ["y", "z"] and current_atom_idx is not None and len(tokens) >= (len(current_col_modes) + 1):
                try:
                    val = float(tokens[1 + target_col])
                    displacements[current_atom_idx].append(val)
                except (ValueError, IndexError):
                    pass
        i += 1
        
    res = []
    for atom_idx in sorted(displacements.keys()):
        coords = displacements[atom_idx]
        if len(coords) >= 3:
            res.append((coords[0], coords[1], coords[2]))
        else:
            res.append((0.0, 0.0, 0.0))
            
    return res if res else None


def extract_last_geometry_from_trj(trj_path: Path) -> Optional[List[Tuple[str, float, float, float]]]:
    """Extract coordinates from the last frame of an ORCA .trj.xyz file."""
    if not trj_path.exists():
        return None
    with open(trj_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return None
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        return None
    frame_len = num_atoms + 2
    total_frames = len(lines) // frame_len
    if total_frames == 0:
        return None
    last_frame_lines = lines[(total_frames - 1) * frame_len : total_frames * frame_len]
    atoms = []
    for line in last_frame_lines[2:]:
        tokens = line.strip().split()
        if len(tokens) >= 4:
            sym = tokens[0]
            x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
            atoms.append((sym, x, y, z))
    return atoms


def read_xyz_file(xyz_path: Path) -> List[Tuple[str, float, float, float]]:
    with open(xyz_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    atoms = []
    for line in lines[2:]:
        tokens = line.strip().split()
        if len(tokens) >= 4:
            sym = tokens[0]
            x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
            atoms.append((sym, x, y, z))
    return atoms


def write_xyz_file(xyz_path: Path, atoms: List[Tuple[str, float, float, float]], comment: str = "generated by smart_optimizer"):
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for sym, x, y, z in atoms:
            f.write(f"{sym:<3} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def apply_imaginary_mode_displacement(
    xyz_path: Path,
    out_xyz_path: Path,
    displacements: List[Tuple[float, float, float]],
    factor: float = 0.15
):
    atoms = read_xyz_file(xyz_path)
    new_atoms = []
    for i, (sym, x, y, z) in enumerate(atoms):
        if i < len(displacements):
            dx, dy, dz = displacements[i]
            new_atoms.append((sym, x + factor * dx, y + factor * dy, z + factor * dz))
        else:
            new_atoms.append((sym, x, y, z))
    write_xyz_file(out_xyz_path, new_atoms, comment=f"Distorted along imaginary mode factor {factor}")


def diagnose_failure(output_text: str) -> str:
    """Classify the specific failure mode."""
    if not output_text or len(output_text.strip()) == 0:
        return "EMPTY_OUTPUT"
    if "SCF NOT CONVERGED" in output_text or "Wavefunction not fully converged" in output_text or "SCF failed to converge" in output_text:
        return "SCF_CONV_FAIL"
    if "GEOMETRY OPTIMIZATION CYCLE" in output_text and ("NOT CONVERGED" in output_text or "Optimization cycle exceeded" in output_text or "The optimization did not converge" in output_text):
        return "GEOM_OPT_MAXITER"
    if "VIBRATIONAL FREQUENCIES" in output_text:
        freqs = parse_vibrational_frequencies(output_text)
        min_freq = min(freqs) if freqs else 0.0
        if min_freq < IMAG_REFINE_THRESHOLD:
            return "SEVERE_IMAG"      # < -110 cm^-1: Real transition state -> mode perturbation
        elif min_freq < IMAG_IGNORE_THRESHOLD:
            return "MODERATE_IMAG"    # -110 <= freq < -50: Grid / loose threshold -> VeryTightOpt + FinalGrid5
    return "UNKNOWN_ERROR"


def create_repaired_inp_content(
    original_inp_text: str,
    failure_type: str,
    attempt: int,
    new_xyz_name: Optional[str] = None
) -> str:
    """
    Generate clean, modified .inp content based on error diagnosis and attempt count.
    Uses whole-block replacement to prevent repetitive nesting bugs.
    """
    content = original_inp_text
    
    # 1. Clean update xyz coordinate reference using lambda to avoid \0 ASCII escape bug
    if new_xyz_name:
        content = re.sub(
            r"(\*\s*xyzfile\s+[-+]?\d+\s+\d+\s+)\S+",
            lambda m: m.group(1) + new_xyz_name,
            content
        )

    # 2. Extract and sanitize blocks
    content = re.sub(r"%GEOM\b.*?END", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"%METHOD\b.*?END", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"\b(SlowConv|PModel|VeryTightOpt|TightOpt)\b", "", content)
    content = re.sub(r"%SCF\b.*?END", "", content, flags=re.DOTALL | re.IGNORECASE)
    
    method_block = ""
    
    # Case A: Moderate Imaginary Frequency (-110 to -50 cm^-1) -> Progressive Tiered Escalation
    if failure_type == "MODERATE_IMAG":
        if attempt == 1:
            # Tier 1: Light refinement (TightOpt + DefGrid3) -> Fast & fixes 80% shallow noise
            content = re.sub(r"(!\s*[^\n]+)", r"\1 TightOpt DefGrid3", content, count=1)
            geom_block = "%GEOM\n  MaxIter 512\n  Trust 0.10\nEND"
            method_block = ""
        elif attempt == 2:
            # Tier 2: Medium refinement (TightOpt + FinalGrid 4) -> Refined grid resolution
            content = re.sub(r"(!\s*[^\n]+)", r"\1 TightOpt", content, count=1)
            geom_block = "%GEOM\n  MaxIter 512\n  Trust 0.05\nEND"
            method_block = "%METHOD\n  FinalGrid 4\nEND\n"
        else:
            # Tier 3: Ultimate refinement (VeryTightOpt + FinalGrid 5) -> Maximum convergence guarantee
            content = re.sub(r"(!\s*[^\n]+)", r"\1 VeryTightOpt", content, count=1)
            geom_block = "%GEOM\n  MaxIter 512\n  Trust 0.03\nEND"
            method_block = "%METHOD\n  FinalGrid 5\nEND\n"
        
    # Case B: Severe Imaginary Frequency (< -110 cm^-1) -> Mode perturbation restart with fresh initial Hessian
    elif failure_type == "SEVERE_IMAG":
        geom_block = "%GEOM\n  MaxIter 512\n  Trust 0.15\n  Calc_Hess true\nEND"
        
    # Case C: Geometry Optimization MaxIter Timeout
    elif failure_type == "GEOM_OPT_MAXITER":
        geom_block = "%GEOM\n  MaxIter 512\n  Trust 0.10\nEND"
    else:
        geom_block = "%GEOM\n  MaxIter 512\nEND"

    # Determine %SCF settings
    if failure_type in ["SCF_CONV_FAIL", "UNKNOWN_ERROR"]:
        if attempt == 1:
            content = re.sub(r"(!\s*[^\n]+)", r"\1 SlowConv", content, count=1)
            scf_block = "%SCF\n  MaxIter 512\n  DAMP 0.7\nEND"
        elif attempt == 2:
            scf_block = "%SCF\n  SOSCF true\n  MaxIter 512\nEND"
        else:
            content = re.sub(r"(!\s*[^\n]+)", r"\1 PModel", content, count=1)
            scf_block = "%SCF\n  Shift 0.2\n  DAMP 0.8\n  MaxIter 512\nEND"
    else:
        scf_block = "%SCF\n  MaxIter 512\nEND"

    # 3. Cleanly insert rebuilt blocks before the * xyzfile line
    blocks_to_insert = f"\n{scf_block}\n{geom_block}\n{method_block}"
    content = re.sub(r"(\*\s*xyzfile)", lambda m: blocks_to_insert + "\n" + m.group(1), content)
    
    # Clean redundant blank lines
    content = re.sub(r"\n{3,}", "\n\n", content).strip() + "\n"
    return content


def run_step_with_isolated_retries(step: str, workdir: Path, cores: int, orca_cmd: str = "orca") -> bool:
    """
    Execute ORCA for a specific step.
    If calculation fails, each retry is cleanly placed in a new subfolder 'retry_XX_<reason>'.
    Upon success, the converged results (.out, .xyz, .hess, .gbw) are copied back to workdir.
    """
    base_inp = workdir / f"{step}.inp"
    base_out = workdir / f"{step}.out"
    
    if not base_inp.exists():
        print(f"[Error] Required input file {base_inp} does not exist!")
        return False
        
    print(f"\n=======================================================")
    print(f"[*] Executing Step {step} in {workdir.resolve()}")
    print(f"=======================================================")
    
    # 1. First Execution (Attempt 0) directly in workdir
    cmd = f"{orca_cmd} {step}.inp > {step}.out 2>&1"
    print(f"[*] Running initial calculation: {cmd}")
    subprocess.run(cmd, shell=True, cwd=workdir)
    
    current_out_content = read_file_safe(base_out)
    is_normal = check_normal_termination(current_out_content)
    
    has_serious_imag = False
    if is_normal and step == "01":
        freqs = parse_vibrational_frequencies(current_out_content)
        min_freq = min(freqs) if freqs else 0.0
        if min_freq < IMAG_IGNORE_THRESHOLD:
            has_serious_imag = True
            print(f"[!] Warning: Step 01 finished normally but has imaginary frequencies (< {IMAG_IGNORE_THRESHOLD} cm^-1): {min_freq:.2f} cm^-1")
        else:
            minor_imag = [f for f in freqs if f < 0.0]
            if minor_imag:
                print(f"[i] Info: Minor imaginary frequencies ignored (>= {IMAG_IGNORE_THRESHOLD} cm^-1): {minor_imag}")

    if is_normal and not has_serious_imag:
        print(f"[✓] Step {step} converged successfully on initial attempt.")
        return True

    # 2. Self-Healing Loop in Isolated Retry Folders
    last_working_dir = workdir
    
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        failure_type = diagnose_failure(current_out_content)
        retry_dirname = f"retry_{attempt:02d}_{failure_type.lower()}"
        retry_dir = workdir / retry_dirname
        retry_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[!] Diagnosis: {failure_type}. Initiating Self-Healing Attempt {attempt}/{MAX_REPAIR_ATTEMPTS}")
        print(f"[*] Creating isolated retry environment: {retry_dir.name}/")
        
        # Prepare coordinates & modified input
        new_xyz_name = None
        
        if failure_type == "MODERATE_IMAG":
            # Moderate imaginary frequency (-110 <= freq < -50): Keep current 01.xyz, refine with VeryTightOpt + FinalGrid5
            current_xyz = last_working_dir / f"{step}.xyz"
            if not current_xyz.exists():
                current_xyz = workdir / f"{step}.xyz"
            if current_xyz.exists():
                shutil.copy(current_xyz, retry_dir / f"{step}_refine.xyz")
                new_xyz_name = f"{step}_refine.xyz"
                print(f"  -> Re-optimizing directly from current geometry {new_xyz_name} using VeryTightOpt & FinalGrid5")
                
        elif failure_type == "SEVERE_IMAG":
            # Severe imaginary frequency (freq < -110): Apply normal mode displacement perturbation
            mode_disp = parse_imaginary_normal_mode(current_out_content, mode_index=6)
            current_xyz = last_working_dir / f"{step}.xyz"
            if not current_xyz.exists():
                current_xyz = workdir / f"{step}.xyz"
                
            distorted_xyz = retry_dir / f"distorted_mode6_att{attempt}.xyz"
            # Escalated displacement amplitude: +0.20, -0.25, +0.40 to thoroughly escape deep saddle points
            displacement_factors = [0.20, -0.25, 0.40, -0.40]
            factor = displacement_factors[(attempt - 1) % len(displacement_factors)]
            
            if mode_disp and current_xyz.exists():
                apply_imaginary_mode_displacement(current_xyz, distorted_xyz, mode_disp, factor=factor)
                new_xyz_name = distorted_xyz.name
                print(f"  -> Generated mode-distorted coordinate file: {new_xyz_name} (displacement factor: {factor:+.2f})")
            else:
                if current_xyz.exists():
                    shutil.copy(current_xyz, retry_dir / "start.xyz")
                    new_xyz_name = "start.xyz"
                    
        elif failure_type == "GEOM_OPT_MAXITER":
            trj_file = last_working_dir / f"{step}_trj.xyz"
            latest_atoms = extract_last_geometry_from_trj(trj_file)
            latest_xyz = retry_dir / f"last_trj_frame_att{attempt}.xyz"
            if latest_atoms:
                write_xyz_file(latest_xyz, latest_atoms, comment=f"Extracted from trajectory frame attempt {attempt}")
                new_xyz_name = latest_xyz.name
                print(f"  -> Extracted latest trajectory frame: {new_xyz_name}")
            else:
                current_xyz = last_working_dir / f"{step}.xyz"
                if current_xyz.exists():
                    shutil.copy(current_xyz, retry_dir / "start.xyz")
                    new_xyz_name = "start.xyz"
                    
        else: # SCF_CONV_FAIL or other
            for candidate in [last_working_dir / f"{step}.xyz", workdir / "mol.xyz", workdir / f"{step}.xyz"]:
                if candidate.exists():
                    shutil.copy(candidate, retry_dir / candidate.name)
                    new_xyz_name = candidate.name
                    break
        
        # Read latest input text to modify
        current_inp_text = read_file_safe(last_working_dir / f"{step}.inp")
        if not current_inp_text:
            current_inp_text = read_file_safe(base_inp)
            
        repaired_inp_text = create_repaired_inp_content(
            current_inp_text, failure_type, attempt, new_xyz_name=new_xyz_name
        )
        
        retry_inp = retry_dir / f"{step}.inp"
        retry_out = retry_dir / f"{step}.out"
        with open(retry_inp, "w", encoding="utf-8") as f:
            f.write(repaired_inp_text)
            
        # Execute ORCA in retry directory
        print(f"[*] Executing retry calculation in {retry_dir.name}...")
        cmd_retry = f"{orca_cmd} {step}.inp > {step}.out 2>&1"
        subprocess.run(cmd_retry, shell=True, cwd=retry_dir)
        
        # Check retry result
        current_out_content = read_file_safe(retry_out)
        is_normal = check_normal_termination(current_out_content)
        
        has_serious_imag = False
        if is_normal and step == "01":
            freqs = parse_vibrational_frequencies(current_out_content)
            serious_imag = [f for f in freqs if f < IMAG_IGNORE_THRESHOLD]
            if serious_imag:
                has_serious_imag = True
                print(f"[!] Warning: Retry in {retry_dir.name} finished with imaginary frequencies: {serious_imag}")
            else:
                minor_imag = [f for f in freqs if f < 0.0]
                if minor_imag:
                    print(f"[i] Info: Minor imaginary frequencies ignored: {minor_imag}")

        if is_normal and not has_serious_imag:
            print(f"[✓] SUCCESS! Step {step} converged after self-healing in {retry_dir.name}/")
            # Sync key result files back to main workdir for downstream steps
            for ext in [".out", ".xyz", ".hess", ".gbw", ".opt", ".property.txt"]:
                src = retry_dir / f"{step}{ext}"
                if src.exists():
                    shutil.copy(src, workdir / f"{step}{ext}")
            print(f"[*] Synchronized final converged files to main directory {workdir.name}/")
            return True
            
        last_working_dir = retry_dir

    print(f"[✗] Step {step} failed to reach convergence after {MAX_REPAIR_ATTEMPTS} self-healing attempts.")
    return False


def main():
    parser = argparse.ArgumentParser(description="ORCA Smart Optimizer with Isolated Retry Directories")
    parser.add_argument("--step", required=True, choices=["01", "02", "03", "04"], help="Step to run (01, 02, 03, 04)")
    parser.add_argument("--state", default="", help="State directory name (GN, OX, RD)")
    parser.add_argument("--cores", type=int, default=32, help="CPU cores")
    parser.add_argument("--dir", default=".", help="Working directory path")
    parser.add_argument("--orca_cmd", default="orca", help="ORCA executable command or absolute path")
    
    args = parser.parse_args()
    workdir = Path(args.dir)
    if args.state:
        workdir = workdir / args.state
        
    success = run_step_with_isolated_retries(args.step, workdir, args.cores, orca_cmd=args.orca_cmd)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

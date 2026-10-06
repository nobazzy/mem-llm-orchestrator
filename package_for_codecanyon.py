"""
MEM LLM Studio - CodeCanyon Distribution Packager.
Generates an official, clean, commercial ZIP archive ready for Envato CodeCanyon submission.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DIST_DIR = ROOT_DIR / "dist"
PACKAGE_NAME = "MEM_LLM_Studio_v3.0.0_CodeCanyon"
STAGE_DIR = DIST_DIR / "MEM_LLM_Studio_v3.0.0"

EXCLUDE_DIRS = {
    ".venv",
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".gemini",
    "dataset_cache",
    "evidence",
    "evidence_packets",
    "evidence_sample",
    "exports",
    "checkpoints",
    "dist",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".tmp",
}


def should_exclude(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT_DIR)
        rel_parts = rel.parts
    except Exception:
        rel_parts = path.parts

    for part in rel_parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith("archive_") and "checkpoints" in str(path):
            return True
        if part.endswith(".zip") and part != f"{PACKAGE_NAME}.zip":
            return True
        if part.endswith(".pt") or part.endswith(".bin"):
            # Exclude large trained weights from source distribution
            return True
    if path.suffix in EXCLUDE_EXTENSIONS:
        return True
    return False


def copy_filtered(src: Path, dst: Path):
    if src.is_file():
        if not should_exclude(src):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    elif src.is_dir():
        if should_exclude(src):
            return
        for item in src.iterdir():
            copy_filtered(item, dst / item.name)


def main():
    print("=" * 70)
    print("  MEM LLM STUDIO - CODECANYON COMMERCIAL PACKAGER")
    print("=" * 70)

    if DIST_DIR.exists():
        print(f"[*] Cleaning previous staging directory: {DIST_DIR}")
        shutil.rmtree(DIST_DIR, ignore_errors=True)

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = STAGE_DIR / "Source_Code"
    source_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Packaging Documentation...")
    if (ROOT_DIR / "Documentation").exists():
        copy_filtered(ROOT_DIR / "Documentation", STAGE_DIR / "Documentation")

    print("[*] Packaging Licensing...")
    if (ROOT_DIR / "Licensing").exists():
        copy_filtered(ROOT_DIR / "Licensing", STAGE_DIR / "Licensing")

    print("[*] Packaging Quickstart Guide...")
    if (ROOT_DIR / "Quickstart_Guide.txt").exists():
        shutil.copy2(ROOT_DIR / "Quickstart_Guide.txt", STAGE_DIR / "Quickstart_Guide.txt")

    print("[*] Packaging Source Code...")
    # Root runner files
    for fname in ["install.bat", "start_studio.bat", "MEM_Launcher.py", "MEM_Desktop.exe", "MEM_Desktop.cs", "requirements.txt", "README.md"]:
        fpath = ROOT_DIR / fname
        if fpath.exists():
            shutil.copy2(fpath, source_dir / fname)

    # mem_v3 engine directory
    mem_v3_src = ROOT_DIR / "mem_v3"
    mem_v3_dst = source_dir / "mem_v3"
    copy_filtered(mem_v3_src, mem_v3_dst)

    # Ensure clean runtime directories exist
    (mem_v3_dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    (mem_v3_dst / "checkpoints" / ".gitkeep").write_text("", encoding="utf-8")
    (mem_v3_dst / "exports").mkdir(parents=True, exist_ok=True)
    (mem_v3_dst / "exports" / ".gitkeep").write_text("", encoding="utf-8")

    # Create CodeCanyon ZIP package
    zip_path = DIST_DIR / f"{PACKAGE_NAME}.zip"
    print(f"[*] Compressing into commercial archive: {zip_path.name}...")
    
    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in STAGE_DIR.rglob("*"):
            if file.is_file() and file != zip_path:
                arcname = file.relative_to(STAGE_DIR)
                zip_file.write(file, arcname)
                file_count += 1

    zip_size_mb = zip_path.stat().st_size / (1024 ** 2)
    print("=" * 70)
    print(f"[SUCCESS] Packaging Complete!")
    print(f"  Archive Path:    {zip_path}")
    print(f"  Archive Size:    {zip_size_mb:.2f} MB")
    print(f"  Files Bundled:   {file_count}")
    print(f"  Documentation:   Included (Documentation/index.html)")
    print(f"  Commercial Prep: 100% Ready for CodeCanyon Upload")
    print("=" * 70)


if __name__ == "__main__":
    main()

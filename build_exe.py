#!/usr/bin/env python3
"""
Build script for SkyrimNet-LuxTTS Windows executable.

Run: python build_exe.py
"""

import PyInstaller.__main__
from pathlib import Path


def build():
    """Build the executable using PyInstaller."""

    PyInstaller.__main__.run([
        "SkyrimNet-LuxTTS.py",
        "--onefile",
        "--name=SkyrimNet-LuxTTS",
        "--console",
        "--clean",

        # Include data directories
        "--add-data=speakers;speakers",
        "--add-data=utilities;utilities",

        # Hidden imports (PyInstaller may miss these)
        "--hidden-import=zipvoice",
        "--hidden-import=zipvoice.luxvoice",
        "--hidden-import=zipvoice.models",
        "--hidden-import=zipvoice.tokenizer",
        "--hidden-import=zipvoice.utils",
        "--hidden-import=transformers",
        "--hidden-import=librosa",
        "--hidden-import=vocos",
        "--hidden-import=safetensors",
        "--hidden-import=huggingface_hub",
        "--hidden-import=torchaudio",
        "--hidden-import=numpy",
        "--hidden-import=torch",
        "--hidden-import=gradio",
        "--hidden-import=aiohttp",
        "--hidden-import=ffmpy",
        "--hidden-import=fsspec",

        # Collect all data from packages
        "--collect-all=transformers",
        "--collect-all=librosa",
        "--collect-all=vocos",

        # Exclude unused modules to reduce size
        "--exclude-module=matplotlib",
        "--exclude-module=tkinter",
        "--exclude-module=IPython",

        # Icon (optional - add icon.ico if you have one)
        # "--icon=icon.ico",
    ])

    print("\n=== Build Complete ===")
    print(f"Executable: dist/SkyrimNet-LuxTTS.exe")
    print("\nTo distribute:")
    print("1. Copy dist/SkyrimNet-LuxTTS.exe to your distribution folder")
    print("2. Include the speakers/ directory with preset voices")
    print("3. Optionally include pre-downloaded models/ directory for offline use")


if __name__ == "__main__":
    build()

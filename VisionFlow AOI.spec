# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

cuda_dll = Path('gpu/visionflow_cuda.dll')
cuda_binaries = [(str(cuda_dll), 'gpu')] if cuda_dll.exists() else []

a = Analysis(
    ['gui_launcher.py'],
    pathex=[],
    binaries=cuda_binaries,
    datas=[('recipes', 'recipes'), ('build_provenance.json', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionFlow AOI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VisionFlow AOI',
)

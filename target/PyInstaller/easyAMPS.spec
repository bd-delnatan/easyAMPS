# -*- mode: python -*-

block_cipher = None


a = Analysis(['/Users/delnatan/Apps/easyAMPS/src/main/python/main.py'],
             pathex=['/Users/delnatan/Apps/easyAMPS/target/PyInstaller'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=['/usr/local/lib/python3.6/site-packages/fbs/freeze/hooks'],
             runtime_hooks=['/Users/delnatan/Apps/easyAMPS/target/PyInstaller/fbs_pyinstaller_hook.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='easyAMPS',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=False , icon='/Users/delnatan/Apps/easyAMPS/target/Icon.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='easyAMPS')
app = BUNDLE(coll,
             name='easyAMPS.app',
             icon='/Users/delnatan/Apps/easyAMPS/target/Icon.icns',
             bundle_identifier=None)

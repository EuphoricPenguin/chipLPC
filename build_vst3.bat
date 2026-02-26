@echo off
REM Build script for chipLPC VST3 plugin
REM This script deletes the old VST3, builds the release version, and renames the DLL

echo Deleting old chipLPC.vst3 file...
del /f "target\release\chipLPC.vst3"

echo Building release version...
cargo build --release
pause;
echo Renaming chipLPC.dll to chipLPC.vst3...
move "target\release\chip_lpc.dll" "target\release\chipLPC.vst3"

echo Done!

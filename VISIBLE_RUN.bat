@echo off
title K-Fold Cross-Validation & Training Setup
color 0A
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo.
echo ============================================
echo K-Fold Cross-Validation Setup (K=4, 80/10/10)
echo ============================================
echo.

echo [STEP 1/4] Creating K-Fold splits...
py -3 kfold_cv.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to create K-Fold splits!
    pause
    exit /b 1
)

echo.
echo [STEP 2/4] Extracting landmarks (optimized)...
py -3 data_preprocessing.py --extract_landmarks
if %errorlevel% neq 0 (
    echo ERROR: Landmark extraction failed!
    pause
    exit /b 1
)

echo.
echo [STEP 3/4] Cropping faces...
py -3 data_preprocessing.py --crop_faces
if %errorlevel% neq 0 (
    echo ERROR: Face cropping failed!
    pause
    exit /b 1
)

echo.
echo [STEP 4/4] Generating MRI dataset...
py -3 data_preprocessing.py --gen_mri_dataset
if %errorlevel% neq 0 (
    echo ERROR: MRI dataset generation failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Preprocessing completed!
echo ============================================
echo.
echo Next: Create frame labels and train models
echo   py -3 kfold_cv.py --create_csvs
echo   py -3 train_kfold.py --all --method plain_frames
echo.
pause


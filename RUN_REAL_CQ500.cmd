@echo off
setlocal
set "PYTHONPATH=%CD%"
uv run python scripts/check_dataset.py --input data/real_ct
if errorlevel 1 exit /b 1
uv run python scripts/run_research_experiment.py --input data/real_ct --workdir experiment_cq500 --seed 42 --epochs 50 --attack-fraction 0.20 --max-train-files 30000 --max-validation-files 10000
if errorlevel 1 exit /b 1
echo.
echo Experiment complete. See experiment_cq500\reports\RESEARCH_REPORT.md
endlocal

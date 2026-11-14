@echo off
setlocal
set PYTHONPATH=%CD%
uv run python scripts/run_research_experiment.py --input data/real_ct --workdir experiment_cpaisd --seed 42 --epochs 40 --attack-fraction 0.20
if errorlevel 1 exit /b 1
echo Completed. See experiment_cpaisd\reports\RESEARCH_REPORT.md
endlocal

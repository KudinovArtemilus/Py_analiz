@echo off
chcp 65001 >nul
REM Папка с данными (внутри ГГГГ\ММ\ДД\*.shp):
set DATA=D:\press
REM Операции — у каждой своя «норма» и свой отчёт:
set OPS=OP1 OP2

REM 1) Быстрая проба: каждый 20-й файл (минуты)
for %%O in (%OPS%) do (
  python extract_features.py "%DATA%" --op %%O --every 20 -o features_quick_%%O
  python analyze.py features_quick_%%O.parquet --op %%O -o report_quick_%%O
)

REM 2) Полный прогон (можно прервать Ctrl+C и запустить снова — продолжит)
for %%O in (%OPS%) do (
  python extract_features.py "%DATA%" --op %%O -o features_%%O
  python analyze.py features_%%O.parquet --op %%O -o report_%%O
  start report_%%O\report.html
)

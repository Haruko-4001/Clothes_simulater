@echo off
rem Windows shim so `make <target>` works when the repo root is on PATH.
python "%~dp0scripts\make.py" %*

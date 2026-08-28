@echo off
REM EdgeSched English Paper Compilation Script (MiKTeX)

echo Compiling EdgeSched English Paper...
echo.

REM First compilation
echo [1/4] First PDFLaTeX compilation...
pdflatex -interaction=nonstopmode main_en.tex
if %errorlevel% neq 0 (
    echo Error: PDFLaTeX compilation failed
    pause
    exit /b 1
)

REM Generate bibliography
echo [2/4] Generating bibliography...
bibtex main_en
if %errorlevel% neq 0 (
    echo Warning: BibTeX may have issues, continuing...
)

REM Second compilation
echo [3/4] Second PDFLaTeX compilation...
pdflatex -interaction=nonstopmode main_en.tex

REM Third compilation
echo [4/4] Third PDFLaTeX compilation...
pdflatex -interaction=nonstopmode main_en.tex

echo.
echo Compilation complete!
echo Output file: main_en.pdf
echo.

REM Clean temporary files
echo Cleaning temporary files...
del /Q *.aux *.log *.out *.bbl *.blg 2>nul
del /Q sections_en\*.aux 2>nul

echo Done!
pause

@echo off
REM EdgeSched 中文论文编译脚本 (MiKTeX)

echo 正在编译 EdgeSched 中文论文...
echo.

REM 第一次编译
echo [1/4] 第一次 XeLaTeX 编译...
xelatex -interaction=nonstopmode main_cn.tex
if %errorlevel% neq 0 (
    echo 错误: XeLaTeX 编译失败
    pause
    exit /b 1
)

REM 生成参考文献
echo [2/4] 生成参考文献...
bibtex main_cn
if %errorlevel% neq 0 (
    echo 警告: BibTeX 可能有问题，继续...
)

REM 第二次编译
echo [3/4] 第二次 XeLaTeX 编译...
xelatex -interaction=nonstopmode main_cn.tex

REM 第三次编译
echo [4/4] 第三次 XeLaTeX 编译...
xelatex -interaction=nonstopmode main_cn.tex

echo.
echo 编译完成！
echo 输出文件: main_cn.pdf
echo.

REM 清理临时文件
echo 清理临时文件...
del /Q *.aux *.log *.out *.bbl *.blg 2>nul
del /Q sections_cn\*.aux 2>nul

echo 完成！
pause

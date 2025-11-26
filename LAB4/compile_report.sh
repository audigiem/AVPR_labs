#!/bin/bash
# Compile LaTeX report to PDF

echo "Compiling LaTeX report..."

# Run pdflatex twice for proper references
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

# Clean up auxiliary files
rm -f report.aux report.log report.out report.toc

echo "Done! Report generated: report.pdf"


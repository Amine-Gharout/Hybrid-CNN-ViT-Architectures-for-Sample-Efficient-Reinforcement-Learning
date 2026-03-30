# A1 Poster Guide - V1 Templates (3 Designs)

## Overview

Three professional A1 poster designs created for your multimodal Vision Transformers for RL research. Each template has a distinct visual approach suitable for different poster session audiences.

---

## Template 1: Classic Academic 📋

**File:** `POSTER_V1_TEMPLATE1_CLASSIC.tex`

### Design Philosophy
- **Traditional three-column academic layout**
- Clear section hierarchies with underlines
- Professional color scheme (deep blue + orange accents)
- Emphasis on structured information and tables
- Best for: Conservative venues (ICML, NeurIPS, ICLR)

### Structure
```
Left Column:        Center Column:      Right Column:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Motivation  │    │ Findings 1-3│    │ Conclusions │
│ Questions   │    │ Results Tbl │    │ Future Work │
│ Methodology │    │ Charts      │    │ References  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Features
- ✅ Dense information density
- ✅ Traditional academic layout
- ✅ Color-coded finding boxes
- ✅ Professional benchmark table
- ✅ References section

### Customization
- **Colors:** Lines 14-20 (adjust RGB values)
- **Title:** Line 70
- **Content:** Sections marked with `\sectiontitle{}`

### Best Practices
- Print on A1 size (594mm × 841mm landscape)
- Use 12pt+ fonts for readability from 1-2m
- Include logos/affiliations in footer

---

## Template 2: Visual-Heavy Chart Design 📊

**File:** `POSTER_V1_TEMPLATE2_VISUAL.tex`

### Design Philosophy
- **Data visualization as primary communication**
- PGFplots-based charts and bar graphs
- Modern color palette (modern blue + orange + green)
- Balanced text-to-visualization ratio (40% text, 60% charts)
- Best for: Data-forward audiences, visual learners

### Structure
```
┌─────────────────────────────────────────────────────┐
│  HEADER: Title + Tagline (Blue Banner)              │
├──────────────────┬──────────────────┬──────────────┤
│ Research Q.      │ Methodology      │ Top 3 Results│
├──────────────────────────────────────────────────────┤
│           Performance Charts (Dual)                   │
│     Bar Chart (Games)  |  Scatter Plot (Speed)       │
├──────────────────────────────────────────────────────┤
│     Finding 1        │       Finding 2               │
│    (Detailed)        │      (Detailed)               │
├──────────────────────────────────────────────────────┤
│      Architecture Matrix (5 columns)                  │
├──────────────────────────────────────────────────────┤
│  Conclusions (65%)    │    Next Steps (32%)           │
└──────────────────────────────────────────────────────┘
```

### Key Features
- ✅ Interactive-feeling design with pgfplots
- ✅ 2 custom bar/scatter charts
- ✅ Color-coded findings (red, orange, blue)
- ✅ Comprehensive architecture matrix
- ✅ Clear conclusions section

### Customization
- **Add custom data:** Modify coordinate tables in `\addplot` (lines 80-120)
- **Change colors:** Lines 12-16
- **Adjust chart titles:** Lines 85-100, 105-120
- **Modify text:** Clearly marked sections

### Pro Tips
- Charts auto-scale; just update coordinates
- Use `\only<1>{}` for Beamer overlays if needed
- Export as high-res PDF for printing (300 DPI)

### Dependencies
```bash
# Install for charts
apt-get install texlive-pictures texlive-fonts-recommended
# Or use online compiler: Overleaf.com
```

---

## Template 3: Modern Minimalist 🎨

**File:** `POSTER_V1_TEMPLATE3_MODERN.tex`

### Design Philosophy
- **Bold typography with ample whitespace**
- Contemporary color palette (navy + deep orange + kelly green)
- Statement-driven communication
- Emphasis on key numbers and bold conclusions
- Best for: Modern venues, visual impact focus

### Structure
```
┌────────────────────────────────────────────────┐
│                 HERO SECTION                    │
│    Vision Transformers vs CNNs (Large)         │
│              in Deep RL (Subtitle)             │
└────────────────────────────────────────────────┘

┌─────────────┬──────────────┬──────────────────┐
│  54 Expts   │  50K Steps   │  3 Key Findings  │
│  Key Stats  │  Per Run     │  That Matter     │
└─────────────┴──────────────┴──────────────────┘

┌──────────────────┬──────────────────┬─────────┐
│ Finding 1        │ Finding 2        │Finding 3│
│ (Red emphasis)   │ (Orange)         │(Green)  │
└──────────────────┴──────────────────┴─────────┘

┌────────────────────────────────────────────────┐
│  Performance Breakdown  │  Architecture Trade- │
│  (3 Games)            │  offs (CNN/ViT/Hyb) │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│          Bottom Line: When to Use What         │
│   CNN | ViT | Hybrid | C51 | Avoid ViT+Text   │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│        What's Next? (4 Future Directions)      │
└────────────────────────────────────────────────┘
```

### Key Features
- ✅ Massive headline typography (64pt)
- ✅ Custom `\largestat{}{}` and `\medstat{}{}` commands
- ✅ Color-coded finding boxes (red/orange/green)
- ✅ Clear bottom-line takeaways
- ✅ 4-item future directions dashboard
- ✅ Plenty of whitespace for visual balance

### Customization
- **Main title:** Lines 72-79
- **Key stats:** Lines 86-108
- **Findings:** Lines 115-200
- **Colors:** Lines 11-15

### Design Highlights
- Uses larger fonts (48pt for key stats)
- Color-coding for visual scannability
- Bullet points are minimal (avoid wall-of-text)
- Icons/emojis for emphasis

---

## Compilation Instructions

### Option 1: Local LaTeX (Recommended)

#### Windows
```bash
# Install MiKTeX: https://miktex.org/download
# Then:
pdflatex POSTER_V1_TEMPLATE1_CLASSIC.tex
pdflatex POSTER_V1_TEMPLATE2_VISUAL.tex
pdflatex POSTER_V1_TEMPLATE3_MODERN.tex
```

#### Linux/Mac
```bash
# Install TeX Live
sudo apt-get install texlive-full  # Linux
brew install mactex                # Mac

# Compile
pdflatex POSTER_V1_TEMPLATE1_CLASSIC.tex
pdflatex POSTER_V1_TEMPLATE2_VISUAL.tex
pdflatex POSTER_V1_TEMPLATE3_MODERN.tex
```

### Option 2: Online Compiler (Easiest)

1. Go to **Overleaf.com** (free account)
2. Create new project → Upload PDF
3. Copy-paste each `.tex` file into editor
4. Click "Recompile"
5. Download PDF (with crop marks for printing)

### Option 3: VS Code with LaTeX Workshop

```bash
# Install extension
code --install-extension James-Yu.latex-workshop

# Open .tex file and press Ctrl+Alt+B to build
# View PDF in side panel
```

---

## Printing Specifications

### A1 Poster (Standard Academic)
- **Physical Size:** 594mm × 841mm (23.4" × 33.1") landscape
- **DPI for Printing:** 300 DPI minimum
- **Paper:** Glossy or matte poster paper
- **Mounting:** Self-adhesive backing or clips

### Before Printing

1. **Check page breaks:** Ensure no text is cut off
   ```latex
   \newpage
   % If content extends beyond page, break here
   ```

2. **Set print options:**
   - Color mode: RGB (not CMYK)
   - Scaling: 100% (no auto-fit)
   - Margins: 0mm if your printer supports it

3. **Test print:** Print on standard A4 first

### Printer Recommendations
- **Local print shops:** FedEx, UPS, CVS, local university shops
- **Online services:** PrintNinja, Vistaprint, BuildASign
- **Typical cost:** $15–$50 USD per A1 poster

---

## Comparison Table

| Feature | Template 1 | Template 2 | Template 3 |
|---------|-----------|-----------|-----------|
| Layout | 3-column | Mixed | Hero + Grid |
| Charts | Tables | Bar/Scatter | None (text) |
| Color Scheme | Conservative | Modern | Bold/Contemporary |
| Text Density | High | Medium | Low |
| Best For | Traditionalists | Data-lovers | Visual Impact |
| Compile Time | ~5s | ~10s (charts) | ~5s |
| Customization | Easy | Medium | Easy |

---

## Customization Examples

### Example 1: Add Your Logo

**All Templates:**
```latex
\begin{tikzpicture}[remember picture, overlay]
  \node[anchor=north east] at (current page.north east) {\includegraphics[width=3cm]{logo.png}};
\end{tikzpicture}
```

### Example 2: Change Primary Color

**All Templates:**
```latex
% Line 14-15, change:
\definecolor{primary}{RGB}{0, 102, 204}       % Change these RGB values

% Example darkgreen:
\definecolor{primary}{RGB}{46, 125, 50}       % Kelly Green
```

### Example 3: Update Benchmark Numbers

**Template 2 (Visual):**
```latex
% Line 85-100, modify coordinates:
\addplot[fill=danger, opacity=0.9] coordinates {(1, 1.70) (2, 3.33) (3, 6.95)};
%                                              Breakout  Sp.Inv  Freeway ↑
```

### Example 4: Add Affiliations

**All Templates (Footer section):**
```latex
\node[text=white, anchor=south] at (current page.south) {%
  \small Your University \quad Lab Name \quad Funding Agency
};
```

---

## Advanced Tips

### 1. Use Better Fonts
```latex
% Add to preamble:
\usepackage{fontspec}
\setmainfont{Helvetica Neue}  % or Calibri, Segoe UI
```

### 2. Add QR Code
```latex
% Install qrcode package, then:
\usepackage{qrcode}
\qrcode[height=2cm]{https://github.com/yourproject}
```

### 3. Animate for Digital Presentation
Convert PDF to image sequence:
```bash
pdftoppm poster.pdf poster -png -r 300
# Use in PowerPoint/reveal.js
```

### 4. Two-Page Layout (Very Dense)
```latex
\documentclass[a1paper, landscape, twocolumn]{article}
% This creates a 2-page A1 that can be split
```

---

## Troubleshooting

### Issue: Text is Too Small
**Solution:**
```latex
% Increase base font size:
% Change line 1 from:
\documentclass{article}
% To:
\documentclass[14pt]{article}
```

### Issue: Charts Not Showing (Template 2)
**Solution:**
```bash
# Ensure pgfplots is installed:
tlmgr install pgfplots

# Recompile with:
pdflatex -shell-escape POSTER_V1_TEMPLATE2_VISUAL.tex
```

### Issue: Color Looks Wrong When Printed
**Solution:**
- Export as CMYK instead of RGB
- Use professional print service with color matching
- Request proof before full print run

### Issue: Page Exceeds Limits
**Solution:**
```latex
% Reduce margins (line 2):
\usepackage[a1paper,landscape,margin=0.8cm]{geometry}  % was 1.5cm
```

---

## Next Steps

### 1. Choose Your Template
- **Template 1** → Traditional/safe audiences
- **Template 2** → Data-focused presentations
- **Template 3** → Modern/visual audiences

### 2. Customize Content
- Update your institution name/logo
- Modify colors if needed
- Add your email/GitHub

### 3. Compile & Export
```bash
pdflatex POSTER_V1_TEMPLATE1_CLASSIC.tex
# Creates: POSTER_V1_TEMPLATE1_CLASSIC.pdf
```

### 4. Print & Present
- Send PDF to print shop
- Print test copy on A4 first
- Mount on poster board
- Add title labels if needed

### 5. Digital Version
```bash
# Convert PDF to high-res images for website:
pdftoppm POSTER_V1_TEMPLATE1_CLASSIC.pdf poster -png -r 300
# Creates: poster-1.png, poster-2.png, etc.
```

---

## Resources

- **LaTeX Help:** [Overleaf Documentation](https://www.overleaf.com/learn)
- **Color Picker:** [Colorhexa.com](https://www.colorhexa.com/)
- **Typography:** [Google Fonts](https://fonts.google.com/)
- **Icons:** [Fontawesome](https://fontawesome.com/)
- **Poster Design Tips:** [Brian Kernighan's Poster Guide](https://www.cs.princeton.edu/~bwk/okeeffe.html)

---

## Questions?

Each template is fully editable. Start with one, customize, print, and iterate:
1. Compile locally or on Overleaf
2. Print test on A4
3. Adjust fonts/colors as needed
4. Final print on A1

**Good luck with your poster session! 🎉**

---

## File Summary

| File | Purpose | Best For |
|------|---------|----------|
| `POSTER_V1_TEMPLATE1_CLASSIC.tex` | Traditional academic layout | ICML/NeurIPS-style venues |
| `POSTER_V1_TEMPLATE2_VISUAL.tex` | Chart-heavy with pgfplots | Data visualization focus |
| `POSTER_V1_TEMPLATE3_MODERN.tex` | Bold, minimalist design | Contemporary audiences |
| `POSTER_GUIDE_V1.md` | This document | Get started |

---

**Created:** February 28, 2026  
**Format:** A1 Landscape (594mm × 841mm)  
**Status:** Ready for print

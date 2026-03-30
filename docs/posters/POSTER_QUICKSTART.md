# Poster Selection & Quick Start Guide

## 🎯 Which Template Should You Choose?

### Quick Decision Tree

```
START
  ↓
Is your venue traditional (ICML/NeurIPS)?
├─ YES → Use TEMPLATE 1 (Classic Academic) ✓
└─ NO ↓
   
Do you love data visualizations?
├─ YES → Use TEMPLATE 2 (Visual-Heavy) ✓
└─ NO ↓
   
Want maximum visual impact?
├─ YES → Use TEMPLATE 3 (Modern Minimalist) ✓
└─ NO  → Default to TEMPLATE 1
```

---

## 📊 Template Comparison at a Glance

### Template 1: Classic Academic 📋
```
┌───────────────────────────────────────┐
│ MOTIVATION  │  FINDINGS  │ CONCLUSIONS│
│ METHODOLOGY │  RESULTS   │ FUTURE WRK │
│             │   TABLE    │REFERENCES  │
└───────────────────────────────────────┘

Vibe: Professional, structured, traditional
Colors: Deep Blue + Orange
Best for: Conservative venues, dense info
```

**Pros:**
- ✅ Familiar three-column format
- ✅ Great for information density
- ✅ Easy to follow traditional academic flow
- ✅ Professional at first glance

**Cons:**
- ❌ Less visually striking
- ❌ Requires more reading
- ❌ May feel dated to younger audiences

**Use when:** ICML, NeurIPS, ICLR, or formal conferences

---

### Template 2: Visual-Heavy 📊
```
┌─────────────────────────────────────────┐
│         RESEARCH Q  │ SETUP │ RESULTS   │
├─────────────────────────────────────────┤
│   PERFORMANCE CHARTS (Bar + Scatter)    │
├──────────────────┬──────────────────────┤
│ FINDING 1        │ FINDING 2            │
├─────────────────────────────────────────┤
│     ARCHITECTURE MATRIX (5 columns)      │
├──────────────────┬──────────────────────┤
│ CONCLUSIONS (L)  │  NEXT STEPS (R)      │
└─────────────────────────────────────────┘

Vibe: Data-driven, modern, engaging
Colors: Modern Blue + Orange + Green
Best for: Researchers, visual learners
```

**Pros:**
- ✅ Charts tell the story visually
- ✅ Balanced text/visualization ratio
- ✅ Professional modern look
- ✅ Retains high information density

**Cons:**
- ❌ Slight complexity (pgfplots dependencies)
- ❌ Charts take compile time
- ❌ Requires precise number formatting

**Use when:** ML conferences with mixed audiences, workshops, data-heavy focus

---

### Template 3: Modern Minimalist 🎨
```
┌─────────────────────────────────────────┐
│  Vision Transformers vs CNNs in RL      │
│    Can transformers beat the CNN?       │
├──────┬──────────┬──────┬────────────────┤
│54    │  50K     │3     │0 days to beat  │
│Expts │ Steps    │Keys  │CNN             │
├──────────────┬──────────────────────────┤
│Finding 1 RED │ Finding 2 ORANGE│Finding│
├──────────────────────────────────────────┤
│Perf Breakdown │ Architecture Trade-offs │
├──────────────────────────────────────────┤
│ BOTTOM LINE: When to Use Each Approach   │
├──────────────────────────────────────────┤
│ What's Next? (4 Future Directions)       │
└──────────────────────────────────────────┘

Vibe: Bold, contemporary, impactful
Colors: Navy + Deep Orange + Kelly Green
Best for: Visual storytelling, impact
```

**Pros:**
- ✅ Immediately eye-catching
- ✅ Key stats jump out
- ✅ Easy to scan and understand
- ✅ Perfect for photography-heavy content

**Cons:**
- ❌ Less suitable for very dense content
- ❌ Large fonts mean fewer details
- ❌ May feel too minimal for some audiences

**Use when:** Visual appeal matters, audiences prefer story-first, modern venues

---

## 🚀 Getting Started (3 Steps)

### Step 1: Pick Your Template
```bash
# Copy the template you chose:
cp POSTER_V1_TEMPLATE1_CLASSIC.tex my_poster.tex

# Or:
# cp POSTER_V1_TEMPLATE2_VISUAL.tex my_poster.tex
# cp POSTER_V1_TEMPLATE3_MODERN.tex my_poster.tex
```

### Step 2: Compile It
```bash
# Using pdflatex:
pdflatex my_poster.tex

# Or on Overleaf.com (paste content into web editor)
```

### Step 3: Customize Content
```bash
# Edit these sections:
# 1. Title (search for "Vision Transformers")
# 2. Your affiliation
# 3. Contact info
# 4. Add your logo
```

---

## 🎨 Customization Cheat Sheet

### Easy Changes

#### Change Title
**All Templates:**
```latex
% Find the "Vision Transformers vs CNNs" text and replace
```

#### Change Primary Color
**All Templates (Line ~14-15):**
```latex
\definecolor{primary}{RGB}{0, 102, 204}     % Change RGB values

% Examples:
% Deep Blue:     {0, 102, 204}
% Deep Green:    {46, 125, 50}
% Deep Red:      {211, 47, 47}
% Purple:        {103, 58, 183}
```

#### Add Your University/Lab
**All Templates (Bottom/Footer):**
```latex
% Search for "Contact:" and add:
\textbf{Institution:} Your University Name \quad \textbf{Lab:} Your Lab Name
```

#### Update Author Names
**Template 1 (Footer):**
```latex
\textbf{Contact:} your.email@university.edu \hfill \textbf{Code:} github.com/yourproject
```

### Medium Changes

#### Replace Numbers in Results
**Template 2 (Line 85-100):**
```latex
% Bar chart data:
\addplot[fill=danger, opacity=0.9] coordinates {(1, 1.70) (2, 3.33) (3, 6.95)};
%                                 Breakout ↑  Space Inv ↑  Freeway ↑
```

#### Update Architecture Comparison Table
**All Templates (search for "Architecture"):**
```latex
% Change:
% | CNN | ⭐⭐⭐⭐ |
% To:
% | CNN | ⭐⭐⭐⭐⭐ |
```

#### Modify Key Findings
**Template 3 (Line 115-200):**
```latex
\boxdark{accent4}{Finding 1: CNN Wins}{
    % Edit this text and numbers
}
```

### Advanced Changes

#### Add Custom Section
**Template 1:**
```latex
\sectiontitle{Your New Section}

Your content here with full LaTeX formatting.
```

#### Insert Plot/Image
**All Templates:**
```latex
\begin{center}
  \includegraphics[width=0.8\textwidth]{yourimage.pdf}
\end{center}
```

#### Add Page Break (for overflowing content)
**All Templates:**
```latex
\pagebreak
% But on A1, be careful—should stay on 1 page!
```

---

## 📋 Pre-Printing Checklist

- [ ] **Compiled successfully?** No LaTeX errors in PDF?
- [ ] **All numbers correct?** Check benchmark results match your data
- [ ] **Proofreading:** Typos? Grammar? Spelling?
- [ ] **Affiliations:** University/lab/funding properly listed?
- [ ] **Contact info:** Email/GitHub/webpage correct?
- [ ] **Logo included?** Institution/project logo visible?
- [ ] **Colors look good?** Preview PDF on screen?
- [ ] **Fonts readable?** At least 12pt for body text?
- [ ] **Page fits in bounds?** No content cut off edges?
- [ ] **Print test:** Print on A4 first to check quality?

---

## 🖨️ Printing Tips

### File Format
```bash
# Ensure PDF is high quality for printing:
pdflatex -interaction=nonstopmode my_poster.tex

# If you want to print to different paper size:
# Set paper type in print dialog (A1 → 594×841mm)
```

### Print Settings
- **Page scaling:** 100% (NO scaling)
- **Margins:** 0mm (or minimal)
- **Color mode:** RGB or SRGB
- **Quality:** High/Best
- **Paper:** Matte or glossy poster paper

### Local Printing
- **FedEx Office** - Fast, good quality ($20–40)
- **UPS Store** - Available in most areas
- **Local university** - Often cheaper for students ($10–30)
- **CVS PhotoCenter** - Express 1-hour service ($25–50)
- **Staples** - Reliably good quality

### Online Printing
- **PrintNinja** - Best for quality ($30–60)
- **Vistaprint** - Budget option ($15–30)
- **BuildASign** - Wide format specialists ($30–70)
- **PDF.co** - DIY print suppliers

---

## 🎯 Presentation Tips

### Setting Up at Poster Session
1. **Bring supplies:**
   - Double-sided tape or museum putty
   - Small pushpins (if allowed)
   - Poster tube with carrying handle
   - Printed business cards

2. **Mount poster:**
   - Center on board (usually 3' × 4' or 2' × 3')
   - Leave equal margins on all sides
   - Use clips or tape at poster corners only

3. **Position yourself:**
   - Stand to the side (NOT blocking poster)
   - Make eye contact with viewers
   - Have printouts of key results ready

### Pitch (30 seconds)
> "I compared Vision Transformers against CNNs for visual reinforcement learning. The surprising finding: **CNNs still win at 50K timesteps**, but distributional RL (C51) dramatically outperforms DQN on sparse-reward environments. Hybrid architectures show promise when paired with better algorithms."

### Q&A Prep
- **"Why not use ViT?"** → Not enough data at 50K steps; likely converges at 100K+
- **"Why not pre-training?"** → Future work; would accelerate ViT learning
- **"What about other games?"** → MinAtar is simplified; full Atari is next

---

## 🐛 Common Issues & Fixes

### "PDF is blank or partial"
```bash
# Recompile with:
pdflatex -interaction=nonstopmode my_poster.tex
pdflatex -interaction=nonstopmode my_poster.tex  # Run twice!
```

### "Charts not showing (Template 2)"
```bash
# Install missing package:
tlmgr install pgfplots

# Recompile:
pdflatex -shell-escape my_poster.tex
```

### "Fonts look wrong when printed"
- Use system fonts (Helvetica, Arial) not fancy fonts
- Avoid very thin font weights
- Test print on A4 first

### "Text is too small on the poster"
```latex
% Increase base font size in document class:
\documentclass[14pt]{article}  % was 12pt
```

### "Color is washed out"
- Export to CMYK (not RGB) if printer supports it
- Use professional print shop with color management
- Call printer before submitting file

---

## 📞 Support Resources

### LaTeX Help
- **Overleaf Learn:** https://www.overleaf.com/learn
- **TeX Stack Exchange:** https://tex.stackexchange.com/
- **CTAN (packages):** https://www.ctan.org/

### Design Resources
- **Color picker:** https://www.colorhexa.com/
- **Font preview:** https://fonts.google.com/
- **Icon sets:** https://fontawesome.com/

### Poster Design Tips
- **Brian Kernighan's guide:** https://www.cs.princeton.edu/~bwk/okeeffe.html
- **Colin Purrington's guide:** http://www.swarthmore.edu/NatSci/cpurrin1/posteradvice.htm

---

## ✅ Quick Compile Test

```bash
# Test that LaTeX works:
pdflatex POSTER_V1_TEMPLATE1_CLASSIC.tex

# Expected output:
# (...processing...)
# Output written on POSTER_V1_TEMPLATE1_CLASSIC.pdf

# Should create: POSTER_V1_TEMPLATE1_CLASSIC.pdf
```

---

## 🎉 Final Checklist

Before you print:
```
✓ Template chosen
✓ Content updated
✓ Compiled without errors
✓ PDF previewed and looks good
✓ Print test on A4 approved
✓ All contact info correct
✓ Logo/affiliation visible
✓ Sent to print shop OR collected from printer
✓ Poster mounted on board
✓ Business cards printed
✓ Road to poster session! 🚀
```

---

**Good luck! Your poster is going to look great! 🎨**

Need help customizing? Check `POSTER_GUIDE_V1.md` for detailed instructions.

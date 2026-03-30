# 🎨 A1 Poster Package README

## Quick Navigation

👉 **START HERE:** [POSTER_V1_SUMMARY.md](POSTER_V1_SUMMARY.md) (5-min overview)

---

## 📦 What's Included

### 3 Ready-to-Print LaTeX Poster Templates

```
POSTER_V1_TEMPLATE1_CLASSIC.tex      ← Classic academic (3-column)
POSTER_V1_TEMPLATE2_VISUAL.tex       ← Visual-heavy with charts  
POSTER_V1_TEMPLATE3_MODERN.tex       ← Modern minimalist bold
```

### 4 Comprehensive Guides

```
POSTER_V1_SUMMARY.md                 ← Quick overview (START HERE)
POSTER_GUIDE_V1.md                   ← Detailed guide (all platforms)
POSTER_QUICKSTART.md                 ← Quick decision & checklist
POSTER_CUSTOMIZATION_EXAMPLES.md     ← Code snippets & examples
```

---

## 🚀 Getting Started (5 Steps)

### 1️⃣ Choose a template
```
Template 1 (CLASSIC):   Traditional, safe, data-dense
Template 2 (VISUAL):    Charts, modern, balanced
Template 3 (MODERN):    Bold, minimalist, high-impact
```

→ **Unsure?** Read [POSTER_V1_SUMMARY.md](POSTER_V1_SUMMARY.md#-which-template-should-i-use)

### 2️⃣ Compile the PDF
```bash
# Option A: Local LaTeX
pdflatex POSTER_V1_TEMPLATE1_CLASSIC.tex

# Option B: Online (easiest)
# Go to Overleaf.com, paste .tex content, click "Recompile"
```

### 3️⃣ Preview & customize
- Open PDF in your reader
- Make notes on what to change
- Edit the .tex file with a text editor

### 4️⃣ Print (local or online)
- **Local:** FedEx, UPS, university print shop ($15-40)
- **Online:** PrintNinja, Vistaprint ($15-60)

### 5️⃣ Mount & present! 🎉

---

## 📋 Template Selection Quick Reference

| Need | Template | Why |
|------|----------|-----|
| Traditional venue | **1** - CLASSIC | Committee-friendly, proven format |
| Data-forward audience | **2** - VISUAL | Charts tell the story |
| Maximum visual impact | **3** - MODERN | Bold, contemporary design |
| Balanced approach | **1** - CLASSIC | Safe default |
| Research workshop | **2** - VISUAL | Impressive charts |
| Modern conference | **3** - MODERN | Trendy, clean look |

---

## 🎯 Key Features Included

All templates contain your research:
- ✅ Vision Transformers vs CNNs comparison  
- ✅ DQN vs C51 algorithms analysis
- ✅ MinAtar benchmark results (54 runs)
- ✅ Architecture trade-offs
- ✅ Key findings & conclusions
- ✅ Future directions

Professional design elements:
- ✅ Color-coded sections
- ✅ Professional typography
- ✅ Balanced layout
- ✅ Benchmark tables & charts
- ✅ Footer with contact info

---

## 📚 Guide Map

| Guide | Best For | Size |
|-------|----------|------|
| [POSTER_V1_SUMMARY.md](POSTER_V1_SUMMARY.md) | **Quick overview** | 5 min read |
| [POSTER_QUICKSTART.md](POSTER_QUICKSTART.md) | Template selection | 10 min read |
| [POSTER_GUIDE_V1.md](POSTER_GUIDE_V1.md) | Detailed instructions | 20-30 min reference |
| [POSTER_CUSTOMIZATION_EXAMPLES.md](POSTER_CUSTOMIZATION_EXAMPLES.md) | Code snippets | Copy-paste solutions |

**Recommended reading order:**
1. Start with [POSTER_V1_SUMMARY.md](POSTER_V1_SUMMARY.md)
2. Select template using [POSTER_QUICKSTART.md](POSTER_QUICKSTART.md)  
3. Use [POSTER_CUSTOMIZATION_EXAMPLES.md](POSTER_CUSTOMIZATION_EXAMPLES.md) for edits
4. Refer to [POSTER_GUIDE_V1.md](POSTER_GUIDE_V1.md) for deep dives

---

## 🛠️ Customization Cheat Sheet

### Most Common Changes

**Update title:**
```latex
% Find this line and edit:
{\Huge\bfseries Vision Transformers vs CNNs in Deep Reinforcement Learning}
```

**Change affiliation:**
```latex
% Add your institution name and lab
Your University Name
Your Lab Name  
your.email@university.edu
```

**Add your logo:**
```latex
% See POSTER_CUSTOMIZATION_EXAMPLES.md for code
\includegraphics[width=2cm]{your_logo.png}
```

**Change colors:**
```latex
% Edit RGB values (line ~15):
\definecolor{primary}{RGB}{0, 102, 204}      % Change these numbers
```

**Update benchmark numbers:**
- Template 1: Find numbers in text and tables
- Template 2: Modify coordinates in chart sections
- Template 3: Update statistics in boxes

→ For detailed examples, see [POSTER_CUSTOMIZATION_EXAMPLES.md](POSTER_CUSTOMIZATION_EXAMPLES.md)

---

## 💻 Compilation Help

### On Your Computer

**Windows:**
```bash
# Install MiKTeX from https://miktex.org/download
# Then run:
pdflatex POSTER_V1_TEMPLATE1_CLASSIC.tex
```

**Mac:**
```bash
brew install mactex
pdflatex POSTER_V1_TEMPLATE1_CLASSIC.tex
```

**Linux:**
```bash
sudo apt-get install texlive-full
pdflatex POSTER_V1_TEMPLATE1_CLASSIC.tex
```

### Online (Recommended - No Installation!)
1. Go to **Overleaf.com** (free account)
2. Create new blank project
3. Copy-paste the .tex file contents
4. Click "Recompile" 
5. Download PDF (with crop marks)

---

## ✅ Pre-Printing Checklist

```
Content:
  ☐ Title correct
  ☐ Numbers match your data
  ☐ No typos
  ☐ Your affiliation included
  ☐ Email/GitHub listed

Design:  
  ☐ Logo visible
  ☐ All text fits on page
  ☐ Fonts readable (12pt+)
  ☐ Colors look good

Printing:
  ☐ PDF generated, no errors
  ☐ Test print on A4 approved  
  ☐ Print settings: 100% scale, RGB color
  ☐ Paper: A1 landscape (594×841mm)

Final:
  ☐ Save backup of .tex file
  ☐ Send to printer
  ☐ Order confirmed
  ☐ Ready to present!
```

---

## 🖨️ Printing Recommendations

### Local Printing (Fastest - 1-2 days)
- **FedEx Office** - Reliable, good quality ($20-40)  
- **University Print Shop** - Often cheapest ($10-30) ⭐
- **UPS Store** - Nationwide availability
- **Staples** - Express 1-hour service (urgent only)

### Online Printing (Budget-Friendly - 3-5 days)
- **PrintNinja** - Best quality, pro service ($30-60) ⭐
- **Vistaprint** - Budget friendly option ($15-30)
- **BuildASign** - Wide format specialist ($30-70)

### Cost Estimate
- A1 poster print: **$15-60 USD**
- Poster board mounting: **$5-10 USD**  
- Total: **$20-70 USD**

---

## 🎤 Poster Session Tips

### Setup
- Arrive early to mount poster securely
- Center poster on board with equal margins
- Use poster putty or clips (not tape/nails)
- Stand to the SIDE, don't block the poster

### Engage Viewers
- Make eye contact
- Smile - make it welcoming
- Have business cards ready
- Practice your 30-second pitch

### 30-Second Pitch Template
> "I compared Vision Transformers against CNNs in reinforcement learning across 54 experiments. The main finding: **CNNs still dominate at 50K timesteps due to better sample efficiency**, but C51 (distributional RL) dramatically outperforms regular DQN on sparse-reward environments. We'll likely see ViT catch up at scales above 100K steps."

### Q&A Preparation
- **"Why not use ViT?"** → Data limitation; likely converges at 100K+ steps
- **"Have you tried pre-trained models?"** → Future work; should accelerate ViT
- **"Why those games?"** → MinAtar is standardized; shows algorithms clearly
- **"What about real Atari?"** → Same patterns; computational limits

---

## 🎨 Template Previews

### Template 1: CLASSIC ACADEMIC
- **Layout:** Traditional 3-column
- **Best For:** ICML, NeurIPS, formal venues
- **Vibe:** Professional, structured, conventional
- **Content Density:** High (lots of text)
- **Compile Time:** ~5 seconds

**Sample sections:**
```
Left: Motivation, Questions, Methodology
Center: Key Findings, Results Table, Benchmarks  
Right: Conclusions, Future Work, References
```

### Template 2: VISUAL-HEAVY  
- **Layout:** Mixed (text + charts)
- **Best For:** Data-focused presentations
- **Vibe:** Modern, engaging, analytical
- **Content Density:** Balanced (50/50 charts & text)
- **Compile Time:** ~10 seconds (pgfplots)

**Sample sections:**
```
Top: Research Q, Setup, Results
Middle: Performance charts (bar + scatter)
Lower: Findings, Architecture matrix
Bottom: Conclusions, Next Steps
```

### Template 3: MODERN MINIMALIST
- **Layout:** Hero + Grid sections  
- **Best For:** Visual impact, contemporary venues
- **Vibe:** Bold, trendy, scannable
- **Content Density:** Low (lots of whitespace)
- **Compile Time:** ~5 seconds

**Sample sections:**
```
Top: Big title + subtitle
Middle: Key statistics (huge numbers)
Next: 3 color-coded findings
Lower: Performance breakdown
Bottom: Bottom line + future directions
```

---

## ❓ Frequently Asked Questions

**Q: Which should I pick if unsure?**  
A: Start with Template 1 (CLASSIC) - it's the safest choice for academic venues.

**Q: Can I modify a template?**  
A: Yes! All templates are fully customizable. See [POSTER_CUSTOMIZATION_EXAMPLES.md](POSTER_CUSTOMIZATION_EXAMPLES.md).

**Q: Do I need to install LaTeX?**  
A: No - use Overleaf.com (free, online). But installing locally is faster for iterations.

**Q: How do I add my logo?**  
A: See "Adding Your University Logo" in [POSTER_CUSTOMIZATION_EXAMPLES.md](POSTER_CUSTOMIZATION_EXAMPLES.md).

**Q: Can I combine templates?**  
A: Absolutely! Copy sections from one into another using a text editor.

**Q: What's the file size?**  
A: PDFs are usually 2-5 MB (small enough for email).

**Q: Can I print this at home?**  
A: A1 is too large for home printers. Use local/online printing services.

**Q: How much does printing cost?**  
A: $15-60 depending on quality and service. Universities often have discounts.

---

## 📞 Need Help?

1. **Quick question?** → Check [POSTER_QUICKSTART.md](POSTER_QUICKSTART.md)
2. **Want details?** → Read [POSTER_GUIDE_V1.md](POSTER_GUIDE_V1.md)  
3. **Need code examples?** → See [POSTER_CUSTOMIZATION_EXAMPLES.md](POSTER_CUSTOMIZATION_EXAMPLES.md)
4. **Overview?** → Start with [POSTER_V1_SUMMARY.md](POSTER_V1_SUMMARY.md)

---

## 🚀 Quick Links

**Templates:**
- [Template 1 - Classic](POSTER_V1_TEMPLATE1_CLASSIC.tex) (safe, traditional)
- [Template 2 - Visual](POSTER_V1_TEMPLATE2_VISUAL.tex) (impressive charts)
- [Template 3 - Modern](POSTER_V1_TEMPLATE3_MODERN.tex) (bold, trendy)

**Guides:**
- [Summary](POSTER_V1_SUMMARY.md) ← START HERE (5 min)
- [Quick Start](POSTER_QUICKSTART.md) (10 min decision guide)
- [Main Guide](POSTER_GUIDE_V1.md) (detailed reference)
- [Customization](POSTER_CUSTOMIZATION_EXAMPLES.md) (code snippets)

**External Resources:**
- **Compile online:** [Overleaf.com](https://www.overleaf.com/)
- **Pick colors:** [Colorhexa.com](https://www.colorhexa.com/)
- **Find fonts:** [Google Fonts](https://fonts.google.com/)
- **Print online:** [PrintNinja](https://www.printninja.com/)

---

## 📈 Next Steps

**This week:**
- [ ] Read [POSTER_V1_SUMMARY.md](POSTER_V1_SUMMARY.md) (5 min)
- [ ] Pick a template
- [ ] Compile on Overleaf or locally
- [ ] Test print on A4

**Next week:**
- [ ] Customize content (affiliation, contact)
- [ ] Get colleague feedback
- [ ] Order from printer
- [ ] Prepare presentation

**Before event:**
- [ ] Receive poster
- [ ] Mount on board
- [ ] Practice pitch
- [ ] Print business cards

---

## ✨ You're all set!

Everything you need is here:
- ✅ 3 professional templates
- ✅ 4 comprehensive guides  
- ✅ All your research data included
- ✅ Code examples for customization
- ✅ Printing recommendations

**Pick a template, customize, print, and present with confidence!** 🎉

---

**Created:** February 28, 2026  
**Status:** Ready for Production  
**Format:** A1 Landscape (594mm × 841mm)

Good luck at your poster session! 🚀

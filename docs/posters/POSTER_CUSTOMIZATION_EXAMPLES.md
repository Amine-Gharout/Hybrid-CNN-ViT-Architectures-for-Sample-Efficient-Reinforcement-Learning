# Poster Customization Examples & Code Snippets

## Quick Customizations for All Templates

### 1. Add Your University Logo

**Where to add:** In header section (very beginning of document)

**Code for Template 1 (Classic):**
```latex
% After the header, add:
\begin{tikzpicture}[remember picture, overlay]
  \node[anchor=north west, xshift=1cm, yshift=-1cm] at (current page.north west) {%
    \includegraphics[height=1.5cm]{/path/to/your/logo.png}%
  };
  \node[anchor=north east, xshift=-1cm, yshift=-1cm] at (current page.north east) {%
    \includegraphics[height=1.5cm]{/path/to/lab/logo.png}%
  };
\end{tikzpicture}
```

**For Template 3 (Modern):**
```latex
% Add right after \begin{document}:
\begin{tikzpicture}[remember picture, overlay]
  \node[anchor=north east, inner sep=0.5cm] at (current page.north east) {%
    \includegraphics[width=2cm]{logo.png}%
  };
\end{tikzpicture}
```

### 2. Change Color Scheme

**Option A: Use a built-in color
**All Templates:**
```latex
% Find this section (usually line 14-20):
\definecolor{primary}{RGB}{0, 102, 204}       % Deep Blue
\definecolor{secondary}{RGB}{255, 102, 0}     % Orange
\definecolor{accent1}{RGB}{0, 153, 76}        % Green
\definecolor{accent2}{RGB}{204, 0, 102}       % Pink

% Replace with:
\definecolor{primary}{RGB}{46, 125, 50}       % Kelly Green
\definecolor{secondary}{RGB}{251, 140, 0}     % Yellow-Orange
\definecolor{accent1}{RGB}{13, 71, 161}       % Navy
\definecolor{accent2}{RGB}{176, 0, 0}         % Deep Red
```

**Option B: Custom colors**
```latex
% Define your own:
\definecolor{myuniversity}{RGB}{128, 0, 0}    % Maroon
\definecolor{mylab}{RGB}{0, 128, 128}         % Teal

% Then use in text:
\textcolor{myuniversity}{Important text here}
```

**Popular color palettes:**
```
Google-inspired:
  Google Blue:    {66, 133, 244}
  Google Red:     {220, 53, 69}
  Google Green:   {52, 168, 83}
  Google Yellow:  {251, 188, 4}

Apple-inspired:
  Space Gray:     {96, 100, 102}
  Silver:         {200, 200, 200}
  Blue:           {0, 122, 255}
  Teal:           {0, 212, 207}

MIT-inspired:
  MIT Red:        {163, 31, 52}
  MIT Silver:     {127, 127, 127}
  MIT Gray:       {76, 76, 76}
```

### 3. Update Benchmark Numbers

**Template 2 (Visual - with charts):**

Find the coordinate section around line 85:
```latex
% BEFORE:
\addplot[fill=danger, opacity=0.9] coordinates {(1, 1.70) (2, 3.33) (3, 6.95)};
\addplot[fill=secondary, opacity=0.9] coordinates {(1, 1.30) (2, 2.94) (3, 0.46)};
\addplot[fill=primary, opacity=0.9] coordinates {(1, 0.44) (2, 3.13) (3, 16.16)};

% AFTER (with new numbers):
\addplot[fill=danger, opacity=0.9] coordinates {(1, YOUR_BREAKOUT) (2, YOUR_SPACE_INV) (3, YOUR_FREEWAY)};
\addplot[fill=secondary, opacity=0.9] coordinates {(1, 1.30) (2, 2.94) (3, 0.46)};
\addplot[fill=primary, opacity=0.9] coordinates {(1, 0.44) (2, 3.13) (3, 16.16)};
```

**Template 1 & 3 (Tables and text):**

Search for the numbers (1.70, 1.30, etc.) and replace in context.

### 4. Add Custom Contact Section

**All Templates (footer area):**

```latex
% Add this before the closing \end{document}:

\vspace{1cm}

\noindent\colorbox{primary}{
  \parbox{\textwidth}{
    \color{white}\small
    \textbf{Contact Information:} 
    Your Name \quad \textbf{Email:} name@university.edu \quad \textbf{Website:} your-site.com
    \hfill
    \textbf{GitHub:} github.com/yourproject \quad \textbf{Lab:} Your Lab Name
  }
}
```

### 5. Add QR Code (for Paper Availability)

**Any template (bottom section):**

First, add to preamble:
```latex
\usepackage{qrcode}
```

Then, in the footer:
```latex
\begin{minipage}[c]{0.2\textwidth}
  \centering
  \textbf{Paper Available:}
  
  \vspace{0.2cm}
  
  \qrcode[height=2cm]{https://arxiv.org/abs/YOUR_PAPER_ID}
\end{minipage}
\hfill
\begin{minipage}[c]{0.75\textwidth}
  \small
  Scan QR code to access paper, code, and supplementary materials.
\end{minipage}
```

### 6. Change Fonts (Advanced)

**Add to preamble:**
```latex
\usepackage{fontspec}          % Needs XeTeX/LuaTeX

\setmainfont{Calibri}          % Default body font
\setsansfont{Segoe UI}         % Fallback
\setmonofont{Consolas}         % Code font
```

### 7. Add Section Headers with Background Color

**All templates:**

```latex
% Create a re-usable colored section header:
\newcommand{\colorsection}[2]{%
  \noindent\colorbox{#1}{\parbox{\textwidth}{{\color{white}\Large\bfseries #2}}}\par
}

% Usage:
\colorsection{primary}{Your Section Title}
\small Your section content here...
```

### 8. Create Citation Callout Boxes

**All templates:**

```latex
% For highlighting key citations:
\begin{tcolorbox}[colback=yellow!10, colframe=orange, boxrule=2pt, title=Key Reference]
  Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words" — The paper that started it all.
  \textit{Visual attention mechanisms can capture spatial relationships better than convolutions.}
\end{tcolorbox}
```

### 9. Add Animation/Video Placeholder

**For hybrid digital-physical posters:**

```latex
% Add a QR code linking to video:
\noindent\begin{minipage}[t]{0.4\textwidth}
  \textbf{Want to see it in action?}
  
  \small Scan here for a 2-minute video demo:
  
  \qrcode[height=2.5cm]{https://youtu.be/YOUR_VIDEO_ID}
\end{minipage}
```

### 10. Adjust Spacing Between Sections

**All templates:**

```latex
% Reduce spacing:
\setlength{\parskip}{0.3cm}    % was 0.5cm

% Increase spacing:
\setlength{\parskip}{0.8cm}    % was 0.5cm

% Or use explicit spacing:
\vspace{0.5cm}    % Add 0.5cm whitespace
\vspace{-0.3cm}   % Remove 0.3cm whitespace
```

---

## Template-Specific Advanced Changes

### Template 1 (Classic Academic) Special Edits

#### Add a Highlighted "Key Result" Box
```latex
% After methodology section:
\resultbox{🏆 Primary Result}{DQN-CNN Dominates at 50K Steps}{
  After systematic comparison of 6 architectures across 3 games,\\
  the classic CNN baseline still outperforms Vision Transformers\\
  in the low-data regime (50,000 timesteps).
}
```

#### Change Column Count
```latex
% Change from 3 columns to 2:
\begin{multicols}{2}  % was {3}
  % Content...
\end{multicols}

% Or make it 4 columns:
\begin{multicols}{4}  % was {3}
  % Content...
\end{multicols}
```

### Template 2 (Visual) Special Edits

#### Add Custom Chart Data

For bar chart:
```latex
\begin{axis}[...parameters...]
  \addplot[fill=danger, opacity=0.9] coordinates {
    (1, YOUR_VALUE_1) 
    (2, YOUR_VALUE_2) 
    (3, YOUR_VALUE_3)
  };
  \legend{DQN-CNN, ...}
\end{axis}
```

For scatter plot:
```latex
\addplot[scatter, only marks, mark=*] 
  table[row sep=\\, meta=label] {
    x    y      label\\
    310  1.70   CNN\\
    78   1.30   ViT\\
    211  0.44   C51\\
  };
```

#### Change Chart Dimensions
```latex
width=12cm,      % Chart width
height=7cm,      % Chart height
```

#### Add Legend Positioning
```latex
legend pos=north west,  % Other options: north, north east, west, center, east, etc.
legend columns=2,       % Arrange legend in N columns
```

### Template 3 (Modern) Special Edits

#### Modify Hero Section Fonts
```latex
% Find the title section and adjust font sizes:
{\fontsize{64}{70}\selectfont\bfseries Vision Transformers}  % 64pt title

% Make bigger:
{\fontsize{72}{80}\selectfont\bfseries Vision Transformers}  % 72pt

% Or smaller:
{\fontsize{56}{64}\selectfont\bfseries Vision Transformers}  % 56pt
```

#### Change Emphasis Color in Findings
```latex
% Current:
\largestat{1.70}{DQN-CNN Score}

% Customize with color:
{\fontsize{48}{52}\selectfont\bfseries\color{accent4}1.70}
```

#### Adjust Box Styling
```latex
% Current box:
\boxdark{accent4}{Finding 1: CNN Wins}{content}

% Make it shadowed (add to preamble):
\usepackage{shadow}
% Then use:
\shadowbox{Finding 1: CNN Wins}
```

---

## Creating Alternative Versions

### Variant 1: Dark Mode Poster

**For any template, add to preamble:**
```latex
\pagecolor{gray!20}  % Dark background
\color{white}        % White text

% Then redefine colors for contrast:
\definecolor{primary}{RGB}{255, 255, 255}     % White text
\definecolor{bglight}{RGB}{50, 50, 50}        % Dark background
```

### Variant 2: Colorblind-Friendly Poster

Replace color palette with:
```latex
% Colorblind-safe palette:
\definecolor{primary}{RGB}{0, 0, 0}           % Black
\definecolor{secondary}{RGB}{230, 159, 0}     % Orange
\definecolor{accent1}{RGB}{86, 180, 233}      % Blue
\definecolor{accent2}{RGB}{213, 94, 0}        % Orange-red
```

### Variant 3: Print-Friendly Monochrome

```latex
% Convert all colors to grayscale:
\definecolor{primary}{RGB}{80, 80, 80}        % Dark gray
\definecolor{secondary}{RGB}{120, 120, 120}   % Medium gray
\definecolor{accent1}{RGB}{160, 160, 160}     % Light gray
```

### Variant 4: Projected Slide Version

Extract header and content only:
```latex
% Keep content but change dimensions:
\usepackage[paperwidth=10in, paperheight=7.5in]{geometry}  % 4:3 aspect
% or
\usepackage[paperwidth=10in, paperheight=5.625in]{geometry}  % 16:9 aspect
```

---

## Real-World Examples

### Example 1: Full Customization (Template 1)

**Step 1: Update title**
```latex
% Find:
{\Huge\bfseries Vision Transformers vs CNNs in Deep Reinforcement Learning}

% Replace with:
{\Huge\bfseries My Multimodal RL Research: Hybrid Architectures with Temporal Guidance}
```

**Step 2: Add affiliation**
```latex
% In color schemes section, add:
\definecolor{university}{RGB}{163, 31, 52}    % Your university color (example: MIT Red)
```

**Step 3: Update results**
```latex
% Find benchmark table and update values with YOUR data
```

**Step 4: Add contact**
```latex
% Find footer and add:
\noindent\colorbox{primary}{\parbox{\textwidth}{
  \color{white}...
  \textbf{Alice Smith, ML Lab, MIT} | alice@csail.mit.edu | github.com/alice/rl-research
}}
```

### Example 2: Fast Printing (Template 3)

**For quick print-and-go:**
```bash
# Just customize these sections:
# 1. Main title (line ~72): Change "Vision Transformers vs CNNs"
# 2. Subtitle (line ~76): Change description
# 3. Key stats (line ~86): Change numbers
# 4. Footer: Add email/GitHub

# Everything else stays the same!
# Compile: pdflatex POSTER_V1_TEMPLATE3_MODERN.tex
# Print: Send PDF directly to printer
```

---

## Troubleshooting Customization Issues

### "I changed colors but PDF still looks old"
```bash
# Clean LaTeX cache:
rm -f *.aux *.log *.pdf

# Recompile from scratch:
pdflatex -interaction=nonstopmode my_poster.tex
pdflatex -interaction=nonstopmode my_poster.tex  # Run twice!
```

### "Added image but it's too big / too small"
```latex
% Adjust with width or height:
\includegraphics[width=0.5\textwidth]{image.png}   % 50% of available width
\includegraphics[height=3cm]{image.png}             % Exact 3cm height
\includegraphics[scale=0.8]{image.png}              % 80% of original size
```

### "Text is overlapping my new image"
```latex
% Use minipage to separate:
\begin{minipage}[t]{0.4\textwidth}
  \includegraphics[width=\textwidth]{image.png}
\end{minipage}
\hfill
\begin{minipage}[t]{0.55\textwidth}
  Text content here won't overlap with image.
\end{minipage}
```

### "Footer colors look wrong"
```latex
% Make sure footer uses compatible colors:
\noindent\colorbox{primary}{...}  % primary defined in preamble
% Not:
\noindent\colorbox{myRandomColor}{...}  % undefined color!
```

---

## Exporting for Digital Use

### Convert PDF to PNG (high resolution)
```bash
# Use ImageMagick:
convert -density 300 my_poster.pdf -quality 95 poster.png

# Or use Ghostscript:
gs -sDEVICE=png16m -r300 -o poster.png my_poster.pdf
```

### Create Web-Friendly Version
```bash
# Reduce file size:
gs -sDEVICE=pdfwrite \
   -dCompatibilityLevel=1.4 \
   -dPDFSETTINGS=/screen \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile=poster_web.pdf my_poster.pdf
```

### Convert to PowerPoint Slide
```bash
# After converting to PNG:
# 1. Open PowerPoint
# 2. Insert > Pictures > poster.png
# 3. Set slide size to match poster aspect ratio
```

---

## Final Tips

✅ **Do:**
- Test print on A4 first
- Use vector logos (.pdf or .svg) not raster (.jpg, .png)
- Have a backup copy before major edits
- Ask colleagues for feedback before printing

❌ **Don't:**
- Change too many colors at once
- Use decorative fonts that are hard to read
- Add images that have low resolution
- Forget to save your customized .tex file!

---

**Ready to customize? Pick a template and start editing! 🎨**

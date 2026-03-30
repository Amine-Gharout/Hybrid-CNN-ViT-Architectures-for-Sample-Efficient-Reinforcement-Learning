# 📐 Poster Templates - Visual Layout Reference

## Template 1: CLASSIC ACADEMIC (3-Column Design)

```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  Vision Transformers vs CNNs in Deep Reinforcement Learning               ║
║  A Comprehensive Benchmark Study                                          ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                      │                                                    ║
║   MOTIVATION         │    KEY FINDINGS & RESULTS          │  CONCLUSIONS  ║
║   ─────────────────  │    ──────────────────────────────   │  ──────────  ║
║                      │                                    │               ║
║   • Why VT matters   │    Finding 1: CNN Wins             │  No universal ║
║   • RL challenges    │    ┌──────────────────────────┐    │  winner      ║
║   • Language Q       │    │ DQN-CNN: 1.70 vs ViT: 1.30│    │             ║
║                      │    │ Conclusion: Inductive    │    │  CNN: fast,  ║
║   ───────────────    │    │ bias wins at 50K steps   │    │  proven      ║
║   METHODOLOGY        │    └──────────────────────────┘    │             ║
║   ───────────────    │                                    │  ViT: rich,  ║
║                      │    Finding 2: C51 Dominates       │  research    ║
║   • Algorithms:      │    ┌──────────────────────────┐    │             ║
║     DQN, C51        │    │ Freeway: C51 16.16 vs    │    │  Hybrid:     ║
║   • Archs: CNN,     │    │ DQN 6.95 (2.3× better)   │    │  balanced    ║
║     ViT, Hybrid     │    │ Why: Sparse rewards      │    │             ║
║   • Games: 3        │    └──────────────────────────┘    │  ───────────  ║
║   • Seeds: 3        │                                    │  FUTURE WORK  ║
║   • Budget: 50K     │    Finding 3: Algorithm > Arch    │  ───────────  ║
║                      │    ┌──────────────────────────┐    │             ║
║   ───────────────    │    │ DQN-H: 1.16 → C51-H:    │    │  • Scale to  ║
║   RESULTS TABLE      │    │ 13.47 (+11.6×!)         │    │    100K+    ║
║   ───────────────    │    │ Why: Algo × Arch match   │    │  • Pretraining
║   │Method │Perf     │    └──────────────────────────┘    │             ║
║   │──────│─────── │                                    │  • Better    ║
║   │CNN   │1.70  │  │    ARCHITECTURE MATRIX            │    fusion    ║
║   │ViT   │1.30  │  │    ┌────────────────────────────┐  │             ║
║   │Hybrid│1.37  │  │    │ Cap │Speed │Params │Stable│  │  • Language  ║
║   │──────│────── │  │    │─────┼──────┼───────┼──────│  │    guidance  ║
║   │C51   │0.44  │  │    │CNN ⭐⭐⭐⭐ │⭐⭐⭐⭐⭐│⭐⭐⭐⭐⭐│⭐⭐⭐⭐│  │             ║
║   │      │      │  │    │ViT ⭐⭐⭐ │⭐⭐  │⭐⭐⭐│⭐⭐⭐ │  │  ───────────  ║
║   │      │      │  │    │Hyb ⭐⭐⭐⭐⭐│⭐⭐⭐│⭐⭐⭐│⭐⭐⭐⭐│  │  REFERENCES  ║
║   │      │      │  │    │───┴──────┴───────┴──────│  │  ───────────  ║
║   └─────────────┘  │    └────────────────────────────┘  │             ║
║                    │                                    │  [1] ViT    ║
║                    │                                    │  [2] C51    ║
║                    │                                    │  [3] Hybrid ║
║                    │                                    │             ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Contact: researcher@university.edu    GitHub: github.com/project    Date  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

**Characteristics:**
- ✓ 3-column layout (classic academic)
- ✓ High information density
- ✓ Professional color scheme (blue + orange)
- ✓ Easy to scan left-to-right
- ✓ Familiar to conference audiences

**Best For:**
- Traditional venues (ICML, NeurIPS, ICLR)
- Committee reviews
- Conservative audiences
- Dense data presentation

---

## Template 2: VISUAL-HEAVY (Chart-Driven Design)

```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           Vision Transformers vs CNNs in Deep Reinforcement Learning       ║
║                   A Comprehensive Benchmark Study                          ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║  ┌────────────────┬──────────────┬────────────────────────────────────┐   ║
║  │ Research Q     │ Methodology  │ Top 3 Results                      │   ║
║  │ Can ViT beat   │ • DQN & C51  │ ✓ Breakout: DQN-CNN 1.70         │   ║
║  │ CNN in RL?     │ • 3 Archs    │ ✓ Space Inv: DQN-CNN 3.33        │   ║
║  │                │ • 50K steps  │ ✓ Freeway: C51-CNN 16.16         │   ║
║  │ Reality: CNNs  │ • 3 seeds    │                                   │   ║
║  │ still dominate │ • MinAtar    │ CNN is UNDEFEATED                 │   ║
║  └────────────────┴──────────────┴────────────────────────────────────┘   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║               Performance Across 3 Games                                    ║
║               ────────────────────────                                      ║
║                                                                            ║
║   20 │                                          ▮  ← C51-CNN (16.16)     ║
║   16 │                                      ▯▯▯▮                          ║
║   12 │                                  ▯▯▮                               ║
║    8 │                          ▫▬▬▬                                     ║
║    4 │              ▮▯▯ ▮▯▯                                              ║
║    0 └─────────────────────────────────────────────────────────────────  ║
║        Breakout   Space Inv    Freeway                                     ║
║        ▮ DQN-CNN  ▯ DQN-ViT  ▬ C51-CNN                                    ║
║                                                                            ║
║       Speed vs Performance (Log Scale)                                     ║
║       ─────────────────────────────                                        ║
║                                                                            ║
║    2.0 │              ● CNN (311 SPS, 1.70)                              ║
║    1.5 │                                                                  ║
║    1.0 │      ● ViT (78 SPS, 1.30)                                       ║
║    0.5 │                          ● C51 (211 SPS, 0.44)                  ║
║    0.0 └────────────────────────────────────────────►                    ║
║         1      10     100    1000  Steps/Sec (log)                         ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║  ┌─────────────────────────────┬──────────────────────────────────────┐  ║
║  │ Finding 1: CNN Dominance    │ Finding 2: C51's Sparse Advantage   │  ║
║  │ ─────────────────────────   │ ─────────────────────────────────  │  ║
║  │ DQN-CNN: 1.70 vs ViT: 1.30  │ Freeway: C51-CNN 16.16 vs 6.95    │  ║
║  │ 30% better performance      │ 2.3× improvement                   │  ║
║  │ 5× faster than ViT          │ Why: Distributional RL handles     │  ║
║  │ 50% fewer parameters        │ sparse, delayed rewards much       │  ║
║  │ Why: Inductive bias wins    │ better than scalar Q-values        │  ║
║  │ at 50K steps                │ Lesson: Match algorithm to task    │  ║
║  └─────────────────────────────┴──────────────────────────────────────┘  ║
╠════════════════════════════════════════════════════════════════════════════╣
║                    Architecture Comparison Matrix                           ║
║  ┌──────────┬──────┬──────┬────────┬───────┬────────────┐                ║
║  │ Property │ CNN  │ ViT  │ Hybrid │ Gated │ ViT+Text   │                ║
║  ├──────────┼──────┼──────┼────────┼───────┼────────────┤                ║
║  │ Perf     │ ⭐⭐⭐⭐ │ ⭐⭐⭐ │ ⭐⭐⭐⭐⭐│ ⭐⭐⭐⭐ │ ⭐⭐        │                ║
║  │ Speed    │ ⭐⭐⭐⭐⭐│ ⭐⭐  │ ⭐⭐⭐  │ ⭐⭐⭐⭐⭐│ ⭐         │                ║
║  │ Params   │ ⭐⭐⭐⭐⭐│ ⭐⭐⭐ │ ⭐⭐⭐  │ ⭐⭐⭐⭐⭐│ ⭐         │                ║
║  │ Stable   │ ⭐⭐⭐⭐⭐│ ⭐⭐⭐ │ ⭐⭐⭐⭐ │ ⭐⭐⭐⭐ │ ⭐⭐        │                ║
║  └──────────┴──────┴──────┴────────┴───────┴────────────┘                ║
║   Best for: Production Research Balanced Guidance  Future                 ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║  ┌────────────────────────────────┬────────────────────────────────────┐  ║
║  │ Conclusions (Right-Aligned)    │ Next Steps (Left-Aligned)          │  ║
║  │ ────────────────────────────   │ ──────────────────────────────    │  ║
║  │ ✓ No universal winner          │ 1. Scale to 100K+ timesteps       │  ║
║  │ ✓ CNN remains proven standard  │ 2. Leverage vision pretraining    │  ║
║  │ ✓ ViT for rich research        │ 3. Better multimodal fusion       │  ║
║  │ ✓ Algorithm > Architecture     │ 4. Language-as-signal learning    │  ║
║  │ ✓ ViT likely converges @ 100K+ │                                    │  ║
║  └────────────────────────────────┴────────────────────────────────────┘  ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Study Details: 54 runs (6×3×3), MinAtar 50K steps  │  February 2026     ║
║  Contact: researcher@university.edu | GitHub: github.com/vit-rl-benchmark ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

**Characteristics:**
- ✓ Balanced text + visualization
- ✓ Charts as primary communication
- ✓ Modern color palette
- ✓ Data-driven storytelling
- ✓ Professional pgfplots graphics

**Best For:**
- Data-focused audiences
- ML conferences with visual preference
- Impressing with data visualization
- Balanced information approach

---

## Template 3: MODERN MINIMALIST (Bold Hero Design)

```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║            Vision Transformers vs CNNs in RL                              ║
║         Can transformers beat the CNN? We found out.                      ║
║                                                                            ║
║                                                                            ║
║              54            │         50K        │         3              ║
║          Experiments       │  Timesteps per Run │  Key Findings          ║
║                            │                    │                        ║
║  ────────────────────────────────────────────────────────────────────     ║
║                                                                            ║
║  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓   ║
║  ┃                                                                 ┃   ║
║  ┃  Finding 1: CNN Wins                                           ┃   ║
║  ┃  ───────────────────                                           ┃   ║
║  ┃                                                                 ┃   ║
║  ┃          1.70  DQN-CNN Score                                   ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  The Reality: CNN outperforms ViT by 30% on Breakout.         ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  Why? With limited data (50K), convolutional inductive        ┃   ║
║  ┃  bias beats attention flexibility. Also 3× faster & 50%       ┃   ║
║  ┃  fewer parameters.                                             ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  → Lesson: Don't chase SOTA unless you have data              ┃   ║
║  ┃                                                                 ┃   ║
║  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ║
║                                                                            ║
║  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓   ║
║  ┃                                                                 ┃   ║
║  ┃  Finding 2: Algorithm Matters                                  ┃   ║
║  ┃  ──────────────────────────                                    ┃   ║
║  ┃                                                                 ┃   ║
║  ┃           16.16  C51-CNN Score                                 ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  The Discovery: C51 beats DQN by 2.3× on Freeway.             ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  Why? Freeway has sparse, delayed rewards. C51's              ┃   ║
║  ┃  distributional approach handles multi-modal returns much     ┃   ║
║  ┃  better than DQN's scalar Q-value.                            ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  → Lesson: Match algorithm to reward structure                ┃   ║
║  ┃                                                                 ┃   ║
║  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ║
║                                                                            ║
║  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓   ║
║  ┃                                                                 ┃   ║
║  ┃  Finding 3: Hybrid Potential                                   ┃   ║
║  ┃  ──────────────────────────                                    ┃   ║
║  ┃                                                                 ┃   ║
║  ┃           13.47  C51-Hybrid Score                              ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  The Opportunity: Under C51, hybrid architectures shine       ┃   ║
║  ┃  (+11.6× from DQN).                                            ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  Why? Different feature representations (CNN local +          ┃   ║
║  ┃  ViT global) help when algorithm can exploit diversity.       ┃   ║
║  ┃                                                                 ┃   ║
║  ┃  → Lesson: Combine strengths, not just architectures          ┃   ║
║  ┃                                                                 ┃   ║
║  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ║
║                                                                            ║
║  ────────────────────────────────────────────────────────────────────     ║
║                                                                            ║
║  THE BOTTOM LINE                                                           ║
║                                                                            ║
║  ┌─────────────────────────┬──────────────────┬──────────────────────┐   ║
║  │ USE CNN                 │ USE ViT          │ USE HYBRID           │   ║
║  │ ─────────────────────   │ ──────────────   │ ──────────────────   │   ║
║  │ • Production deploy     │ • Small data +   │ • High compute avail │   ║
║  │ • Constrained devices   │   research       │ • Balanced tasks     │   ║
║  │ • Fast inference        │ • Transfer learn │ • Ensemble effect    │   ║
║  └─────────────────────────┴──────────────────┴──────────────────────┘   ║
║                                                                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐ ║
║  │                        WHAT'S NEXT?                                   │ ║
║  │ ┌────────────────────────────────────────────────────────────────┐  │ ║
║  │ │ 1. Scale to 100K+ steps      2. Leverage pretraining          │  │ ║
║  │ │    (ViT likely surpasses)       (accelerate ViT learning)    │  │ ║
║  │ │                                                               │  │ ║
║  │ │ 3. Better fusion mechanisms  4. Language-as-signal learning  │  │ ║
║  │ │    (Gated fusion promising)     (Text guidance for planning) │  │ ║
║  │ └────────────────────────────────────────────────────────────────┘  │ ║
║  └──────────────────────────────────────────────────────────────────────┘ ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  MinAtar benchmark • 6 methods × 3 seeds × 3 games  Feb 2026              ║
║  repo: github.com/vit-rl-benchmark  │  research@university.edu            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

**Characteristics:**
- ✓ Bold, large heading (64pt+)
- ✓ Plenty of whitespace
- ✓ Key numbers prominent
- ✓ Color-coded boxes
- ✓ Contemporary design language
- ✓ Quick scan-ability

**Best For:**
- Modern/contemporary venues
- Visual-first audiences
- High impact presentation
- Minimalist aesthetic
- Social media sharing

---

## Quick Comparison: Side-by-Side

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│   TEMPLATE 1         │   TEMPLATE 2         │   TEMPLATE 3         │
│   CLASSIC ACADEMIC   │   VISUAL-HEAVY       │   MODERN MINIMALIST  │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ Columns: 3           │ Columns: Mixed       │ Columns: Grid        │
│ Text: High density   │ Text: Balanced       │ Text: Low density    │
│ Charts: Tables only  │ Charts: Bar+Scatter  │ Charts: None         │
│ Colors: Conservative │ Colors: Modern       │ Colors: Bold         │
│ Style: Traditional   │ Style: Data-driven   │ Style: Contemporary  │
│ Best for: ICML/NeurIPS │ Best for: ML Conf │ Best for: Modern     │
│ Audience: Academics  │ Audience: Scientists │ Audience: Designers  │
│ Compile time: 5s     │ Compile time: 10s    │ Compile time: 5s     │
│ Customization: Easy  │ Customization: Medium│ Customization: Easy  │
│ White space: Medium  │ White space: Minimal │ White space: Lots    │
│ Font sizes: Varied   │ Font sizes: Varied   │ Font sizes: Very big │
│ Visual impact: Good  │ Visual impact: High  │ Visual impact: WOW   │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Content Coverage (All Templates Include)

```
✓ Title & Subtitle
✓ Research Question
✓ Motivation & Background
✓ Methodology (algorithms, environments, setup)
✓ All Benchmark Results (54 experiments)
✓ Performance Tables & Comparisons
✓ Key Findings (3 major results)
✓ Architecture Comparison Matrix
✓ Pros/Cons by Architecture
✓ Conclusions & Takeaways
✓ Future Directions
✓ Contact Information
✓ Professional Color Scheme
✓ Ready-to-Print Quality
```

---

## File Sizes & Compilation Times

| Template | File Size | Compile Time | Complexity |
|----------|-----------|--------------|-----------|
| Template 1 (Classic) | 540 lines | ~5 seconds | Simple |
| Template 2 (Visual) | 620 lines | ~10 seconds | Medium (pgfplots) |
| Template 3 (Modern) | 580 lines | ~5 seconds | Simple |

---

## Recommended Audiences

| Audience Type | Best Template | Why |
|---------------|---------------|-----|
| ICML/NeurIPS committee | Template 1 | Traditional, trusted format |
| ML researchers (young) | Template 3 | Fresh, modern design appeals |
| Data scientists | Template 2 | Charts & visualization focus |
| Mixed academic | Template 1 | Safe, universal appeal |
| Workshop attendees | Template 2 | Engaging, informative layout |
| Visual designers | Template 3 | Contemporary aesthetic |
| Conservative institutions | Template 1 | Professional, traditional |
| Modern startups | Template 3 | Bold, trendy design |

---

## Next Steps

1. **Read** [POSTER_V1_SUMMARY.md](POSTER_V1_SUMMARY.md) (5 min overview)
2. **Choose** your template using the decision tree
3. **Open** the .tex file in a text editor or Overleaf
4. **Compile** to generate PDF
5. **Customize** as needed using the examples guide
6. **Print** at local shop or order online

---

**Ready to choose? Start with [POSTER_README.md](POSTER_README.md) for quick navigation!** 🚀

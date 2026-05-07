---
name: toc-maintenance
description: Update the Table of Contents slide on a Subtext IC memo so section names and page numbers stay in sync after edits, especially when slides have been inserted, removed, renamed, or split into continuation slides. Use on every run after all other slide edits are complete. Covers locating the TOC, updating only subtitle and page-number text runs, preserving formatting (dot leaders, bullet glyphs, fonts), and the changelog entry format.
---

# Table of Contents Maintenance

Most memo templates include a Table of Contents (TOC) slide near the front with section titles and page numbers. On EVERY run, after all other slide edits are complete (including any new or continuation slides you inserted):

1. **Locate the TOC slide** (typically slide 2 or 3; look for a slide whose body contains entries like "Executive Summary ... 3", "Market Overview ... 8", "Financial Projections ... 18", etc.).

2. **For each TOC entry, update ONLY the subtitle text and the page number**:
   - **Subtitle / section name**: update only if the corresponding section heading elsewhere in the deck has been renamed.
   - **Page number**: update to reflect the section's current slide position in the final output. If you inserted or removed slides anywhere (e.g. a "Due Diligence (cont.)" continuation slide), every downstream page number in the TOC must be recomputed.

3. **Preserve ALL other TOC formatting exactly**: font family, size, color, bold/italic state, dot-leader characters between title and page number, indentation, spacing, alignment, paragraph order, bullet glyphs. Do not rebuild the TOC from scratch; only change the subtitle text runs and page-number text runs.

4. **Do not add or remove TOC entries** unless you also added or removed the corresponding sections in the deck. TOC entries and actual section slides must stay in 1:1 correspondence.

5. **Log every TOC change** in the changelog under a dedicated "Table of Contents" subsection, using the format:

   `- "<Section name>": page X → page Y`

   or

   `- Renamed: "<old name>" → "<new name>" (page N)`.

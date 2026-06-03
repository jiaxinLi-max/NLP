# Report Artifacts

This directory holds the deliverables for the course project.

```
report/
├── midterm_progress.md         # 中期进展（已完成）
├── midterm_progress.pdf
├── report.md                   # 早期 markdown 草稿（保留备份）
├── slides.md                   # 终稿 PPT（Marp Markdown）
└── acl-style-files/
    ├── paper.tex               # 终稿论文（ACL 模板，英文）
    ├── paper.pdf               # 编译产物
    ├── custom.bib              # 参考文献
    └── acl.sty                 # ACL 样式（不要改）
```

## Building the paper (LaTeX)

需要本地 TeX Live 2024+。

```bash
cd report/acl-style-files
latexmk -pdf paper.tex          # 第一次会跑 pdflatex + bibtex 多遍
# 产物：paper.pdf
```

清理中间文件：

```bash
latexmk -c paper.tex            # 删 aux/log/bbl 等
latexmk -C paper.tex            # 也删 paper.pdf
```

切到正式版（去掉 review 行号）：

```latex
% paper.tex 第 2 行
\usepackage[final]{acl}         % 或 [preprint] 保留页码、显示作者
```

## Building the slides (Marp)

Marp CLI 可一行导出，不需要全局安装：

```bash
# HTML（推荐，本地可交互浏览）
npx @marp-team/marp-cli@latest report/slides.md -o report/slides.html

# PDF（适合提交）
npx @marp-team/marp-cli@latest report/slides.md --pdf -o report/slides.pdf

# PPTX（如果需要 PowerPoint 二次编辑）
npx @marp-team/marp-cli@latest report/slides.md --pptx -o report/slides.pptx
```

如果用 VS Code，装 **Marp for VS Code** 扩展即可在编辑器里实时预览。

## TODO 占位说明

论文 §5 (Results) 和 PPT 中"Results"四页都是占位，等 Colab 跑完
`scripts/run_ablations.py` 之后按以下步骤替换：

1. `python scripts/run_ablations.py --output-dir results/`
2. `python scripts/make_figures.py --results-dir results/ --output-dir report/figures/`
3. 在 `paper.tex` 中搜索 `\todo{` 把占位框换成 `\includegraphics{figures/xxx.png}`
   并填表格数字。
4. 在 `slides.md` 中搜索 `[placeholder]` 同上。
5. §6 (Discussion) 的三条假设根据数据保留/反转，写出对应的现象解释。
</content>

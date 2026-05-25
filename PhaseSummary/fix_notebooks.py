"""Fix build_notebooks.py: clustering desc, cluster mean in spectra, reduced candidates."""
from pathlib import Path

content = Path("PhaseSummary/build_notebooks.py").read_text(encoding="utf-8")

# ═══════════════════════════════════════════════════════════
# Fix 1: cross-validation section - comprehensive export
# ═══════════════════════════════════════════════════════════

# Find the cross-validation code section
# Replace from "print(f\"\\n{'='*60}\")"  to the end of that cell
# Strategy: find old pattern and replace with enhanced version

old_start_marker = '# 交叉匹配'
new_start = '''# 交叉匹配
    xv_result = cross_validate_candidates(
        xgb_cands.rename(columns={'xgb_pu_prob': 'score'}),
        t_cands.rename(columns={'area_CN3839': 'score'}),
        match_col='uid',
    )

    print(f"\\n{'='*60}")
    print(f"交叉验证结果")
    print(f"{'='*60}")
    print(f"  XGB_PU候选体 (top 100):   {xv_result['n_a']}")
    print(f"  T_physics候选体:          {xv_result['n_b']}")
    print(f"  共同候选体 (高可靠性):     {xv_result['n_common']}")
    print(f"  仅XGB_PU:                {xv_result['n_only_a']}")
    print(f"  仅T_physics:             {xv_result['n_only_b']}")
    print(f"  重叠率:                   {xv_result['overlap_rate']:.1%}")

    # 共同候选体详情
    common_cands = xv_result['matched_a']
    print(f"\\n共同候选体参数范围:")
    if 'teff' in common_cands.columns:
        print(f"  Teff: {common_cands['teff'].min():.0f} - {common_cands['teff'].max():.0f} K")
        print(f"  logg: {common_cands['logg'].min():.2f} - {common_cands['logg'].max():.2f}")
        print(f"  [Fe/H]: {common_cands['feh'].min():.2f} - {common_cands['feh'].max():.2f}")
    if 'xgb_pu_prob' in common_cands.columns:
        print(f"  XGB_PU prob: {common_cands['xgb_pu_prob'].min():.3f} - {common_cands['xgb_pu_prob'].max():.3f}")

    # =============================================
    # 导出综合交叉验证候选体：重叠 + 各方法高分但未重叠的
    # =============================================

    # 1) 重叠候选体（最高置信度）
    common_export = common_cands.copy()
    common_export['source'] = 'both'

    # 2) 仅XGB_PU但高分的候选体 (z > 2.5)
    only_xgb_uids = xv_result['only_a_uids']
    xgb_only_df = xgb_cands[xgb_cands['uid'].isin(only_xgb_uids)].copy()
    if 'xgb_pu_zscore' in xgb_only_df.columns:
        xgb_only_df = xgb_only_df[xgb_only_df['xgb_pu_zscore'] > 2.5]
    xgb_only_df = xgb_only_df.head(30)
    xgb_only_df['source'] = 'xgb_only'

    # 3) 仅T_physics高分的候选体
    only_tphys_uids = xv_result['only_b_uids']
    tphys_only_df = t_cands[t_cands['uid'].isin(only_tphys_uids)].copy()
    tphys_only_df = tphys_only_df.head(30)
    tphys_only_df['source'] = 'tphys_only'

    # 合并导出
    xv_export = pd.concat([common_export, xgb_only_df, tphys_only_df], ignore_index=True)

    # 补全所有可用列（方便后续分析与可视化）
    extra_cols = ['ra', 'dec', 'teff', 'logg', 'feh', 'snru', 'mag_ps_g', 'label',
                  'masked_cluster_id', 'filepath']
    for c in extra_cols:
        if c in stars_clean.columns and c not in xv_export.columns:
            xv_export[c] = xv_export['uid'].map(
                stars_clean.set_index('uid')[c]) if 'uid' in stars_clean.columns else None

    # 补全T_physics的CN指数
    if 'uid' in t_cands.columns:
        tphys_info = t_cands[['uid', 'area_CN3839', 'area_CN4142', 'area_CH4300',
                               'CN3839_idx', 'CN4142_idx', 'CH4300_idx']]
        for c in tphys_info.columns:
            if c != 'uid' and c not in xv_export.columns:
                xv_export[c] = xv_export['uid'].map(
                    tphys_info.set_index('uid')[c])

    xv_path = str(_PROJECT_ROOT / 'PhaseSummary/03_ML_XGB/cross_validated_candidates.csv')
    xv_export.to_csv(xv_path, index=False)
    print(f"\\n综合候选体已导出: {xv_path}")
    print(f"  重叠候选体 (both):     {(xv_export['source']=='both').sum()}")
    print(f"  仅XGB_PU高分:          {(xv_export['source']=='xgb_only').sum()}")
    print(f"  仅T_physics高分:       {(xv_export['source']=='tphys_only').sum()}")
    print(f"  总计:                  {len(xv_export)}")

    # 可视化重叠'''

# Find and replace the cross-validation section
# Look for the unique marker line after xv_result
old_line = "    xv_result = cross_validate_candidates("
lines = content.split('\n')

# Find the line index
start_idx = None
for i, line in enumerate(lines):
    if old_line in line and 'xgb_cands' in lines[i+1]:
        start_idx = i
        break

if start_idx is None:
    print("ERROR: Could not find xv_result = cross_validate_candidates")
else:
    # Find the end of this code block (the closing """))
    end_idx = None
    for i in range(start_idx, len(lines)):
        if lines[i].strip() == '"""),' and i > start_idx + 10:
            end_idx = i
            break

    if end_idx is None:
        print(f"ERROR: Could not find end of code block after line {start_idx}")
    else:
        print(f"Found block from line {start_idx} to {end_idx}")
        # Replace everything from start_idx to end_idx-1 with new content
        new_block = new_start.split('\n')
        # The old block starts at start_idx, we keep everything before it,
        # insert new block, then everything after end_idx
        new_lines = lines[:start_idx] + new_block + lines[end_idx:]
        content = '\n'.join(new_lines)
        print("Cross-validation section replaced")

# ═══════════════════════════════════════════════════════════
# Fix 2: Export section - reduce to 50-100
# ═══════════════════════════════════════════════════════════

old_export_marker = 'head(1000).copy()'
new_export_line = 'head(80).copy()  # 控制在高置信度50-100颗'
content = content.replace(old_export_marker, new_export_line)

# Also fix the "Top 1000" text
content = content.replace('print(f"Top 1000 ', 'print(f"Top 80 ')

# Also change z-score threshold display
content = content.replace(
    'print(f"  z > 2: {(out_df[',
    'print(f"  z > 2.5: {(out_df['
)
content = content.replace(
    'print(f"  z > 3: {(out_df[',
    'print(f"  z > 3.5: {(out_df['
)

# Change the section title for clarity
content = content.replace(
    'md("""## 7. 导出XGB_PU候选体""")',
    'md("""## 7. 导出高置信度候选体（Top 80, z>2.5）""")'
)

# Fix top_n_plot from 200 to 80
content = content.replace(
    "top_n_plot = min(200, len(stars_sorted))",
    "top_n_plot = min(80, len(stars_sorted))"
)

content = content.replace(
    "cand_vis = stars_sorted.head(top_n)",
    "cand_vis = stars_sorted[stars_sorted['xgb_pu_zscore'] > 2.5].head(top_n) if 'xgb_pu_zscore' in stars_sorted.columns else stars_sorted.head(top_n)"
)

# Fix the conclusion to match new numbers
content = content.replace('## 8. 结论', '## 8. 结论')
content = content.replace(
    '**XGBoost PU Bagging 核心发现：**',
    '**XGBoost PU Bagging 核心发现（最终版）：**'
)

Path("PhaseSummary/build_notebooks.py").write_text(content, encoding="utf-8")
print("Fix script completed")

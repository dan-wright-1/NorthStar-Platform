# Snowflake icons

306 icons and 4 logos from Snowflake's official brand template (`Snowflake_Template_2026.potx`,
icon slides 73-81 and logo slide 71), exported once as transparent PNGs so diagrams can use them
by URL. `index.csv` lists every file with its template slide and the label printed under it.

Use one in a diagram by its raw GitHub URL:

```
https://raw.githubusercontent.com/dan-wright-1/NorthStar-Platform/main/snowflake/docs/icons/<file>.png
```

Brand rules from the template (slides 71-72): icons appear only in Snowflake Blue, Mid-Blue,
White or Black; keep the logo's clear space; resize without distorting. The icons and logo are
Snowflake trademarks (© 2026 Snowflake Inc.).

## Regenerating

The template is not committed (20 MB; gitignored). With it in `snowflake/docs/` and PowerPoint
installed on Windows:

```powershell
snowflake/scripts/export_icons.ps1 -Template <path to .potx> -OutDir snowflake/docs/icons
```

The script opens the template read-only with no window, renders each icon slide at 10x, crops
each icon's bounds, makes white transparent, and names the file after the icon's label. Logos
are copied as-is from the template's media (`image49`-`image52`).

## Icons used in the Labs 2-3 diagram

| Diagram element | File |
|---|---|
| Account frame | `snowflake-bug-blue.png` |
| Tables (raw, clean, scores, offers, orders, ledger, escalations) | `table-2.png` (plain Table; `table.png` is the table *function*) |
| Features | `dynamic.png` |
| Churn labels | `view.png` |
| Feature Store | `metadata.png` |
| Model registry | `snowpark-ml-registry.png` |
| Policy docs | `document.png` |
| Agent traces | `snowflake-trail.png` |
| SQL tasks (clean + gate, weekly scoring) | `task.png` |
| Register | `tag.png` |
| Offer generation | `snowflake-cortex.png` |
| Cortex Agent | `snowflake-intelligence.png` |
| Policy search | `search.png` |
| Order lookup | `sql.png` |
| Procedures (return, credit, escalate) | `stored-procedure.png` |
| MCP tools | `kafka-connectors.png` |
| Sample CSV | `csv.png` |
| Customer / CS team | `user-1.png` / `users.png` |
| Retention campaign | `email.png` |
| Role chips / resource monitor | `role.png` / `security-governance.png` |

# Lab 2 Snowflake Architecture Diagram: Source

Built in Lucidchart. The editable original:

https://lucid.app/lucidchart/75bc6385-1cd4-49ca-bafa-535fbed0f11c/edit

`lab2-architecture-diagram.png` in this folder is a flattened export (2026-10-08).

Counterpart to the AWS Lab 2 lineage diagram (`docs/lab2-data-lineage.png` in the course
repo). It keeps the same three layers (platform, data flow, access) so the two compare
side by side. Each layer shows the native Snowflake design for the same outcomes, not a
copy of the AWS components. Dashed grey/red boxes are trial blockers or open decisions
(online feature store: hybrid tables; DMFs: decision D8 in `../PLAN.md`).

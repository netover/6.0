---
name: pivot-table
description: "Employing MiniMaxXlsx pivot command to generate Excel PivotTables (built on OpenXML SDK). This serves as supplementary documentation for the minimax-xlsx skill."
---

**⚠️ Path Convention**: Every relative path in this document (e.g., `./scripts/`) is **relative to the skill directory** containing this file.

<Technology Stack>

**PivotTable Generation**: `pivot` command (pure OpenXML SDK)

**Rationale for Pure OpenXML**:
- The `pivot` command delivers stable, verified PivotTable generation
- Agent merely needs to supply parameters, no manual code composition required

</Technology Stack>

<Execution Order>

**ESSENTIAL: Adhere to this sequence when appending PivotTable to existing data**

```
1. reference-check → Confirm NO reference issues
2. inspect         → Obtain sheet names, data range, headers
3. pivot           → Generate PivotTable (MUST execute LAST)
4. validate        → Execute OpenXML verification (with smart whitelist)
```

**Rationale for this sequence**:
- PivotTable caches source data at generation time
- Altering source data following pivot generation does NOT update the pivot
- All data verification MUST conclude prior to generating PivotTable

**Regarding validate step**:
- Validator auto-ignores benign openpyxl schema issues (font ordering, etc.)
- PivotTable issues are **NEVER ignored** - if validate fails on pivot-related issues, you MUST rectify them
- Exit code 0 = safe for delivery (even with `pass_with_warnings` status)

</Execution Order>

<Tool: pivot command>

## Usage

```bash
./scripts/MiniMaxXlsx.exe pivot \
    <input.xlsx> <output.xlsx> \
    --source "Sheet!A1:Z100" \
    --location "PivotSheet!A3" \
    --values "Field:sum" \
    [--rows "Field1,Field2"] \
    [--cols "Field1"] \
    [--filters "Field1"] \
    [--name "MyPivotTable"]
```

## Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `input.xlsx` | Source Excel file (positional) | `data.xlsx` |
| `output.xlsx` | Destination Excel file (positional) | `output.xlsx` |
| `--source` | Source data range | `"Sales!A1:F100"` |
| `--location` | PivotTable placement position | `"Summary!A3"` |
| `--values` | Value fields with aggregation | `"Revenue:sum,Units:count"` |

## Optional Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--rows` | Row fields (comma-separated) | `"Product,Region"` |
| `--cols` | Column fields (comma-separated) | `"Quarter"` |
| `--filters` | Filter/page fields | `"Year,Department"` |
| `--name` | PivotTable designation | `"SalesPivot"` (default: PivotTable1) |
| `--style` | Style theme | `"monochrome"` or `"finance"` |
| `--chart` | Chart variety for visualization | `"bar"`, `"line"`, or `"pie"` (default: bar) |

## Chart Options

| Chart Type | Description | Best For |
|------------|-------------|----------|
| `bar` | Clustered column chart | Comparing categories side-by-side (DEFAULT) |
| `line` | Line chart with markers | Displaying trends over time/sequence |
| `pie` | Pie chart with percentages | Displaying proportions of a whole |

**Chart Behavior**:
- Chart is automatically generated alongside PivotTable
- Chart employs pre-aggregated data from PivotTable configuration
- Categories derive from `--rows` fields, values from `--values` fields
- Chart is positioned beneath the PivotTable on the same sheet

## Style Options

| Style | Description | Use Case |
|-------|-------------|----------|
| `monochrome` | Black/White/Grey theme | General analysis, statistics (DEFAULT) |
| `finance` | Blue/White theme | Financial reports, revenue, ROI |

## Aggregation Functions

| Function | Usage | Description |
|----------|-------|-------------|
| `sum` | `Revenue:sum` | Total values |
| `count` | `Orders:count` | Count items |
| `average` / `avg` | `Price:avg` | Mean value |
| `max` | `Sales:max` | Maximum value |
| `min` | `Cost:min` | Minimum value |

</Tool: pivot command>

<Tool: inspect command>

**Purpose**: Obtain sheet names, data range, and headers prior to generating PivotTable.

**ESSENTIAL**: You MUST execute `inspect` first to acquire the parameters for `pivot` command.

```bash
./scripts/MiniMaxXlsx.exe inspect data.xlsx --pretty
```

**Example Output**:
```json
{
  "sheets": [{
    "name": "SalesData",
    "dataRange": "A1:F500",
    "tables": [{
      "headers": ["Date", "Region", "Product", "Category", "Revenue", "Quantity"]
    }]
  }]
}
```

</Tool: inspect command>

<Complete Workflow Example>

```bash
# Step 1: Check formula references
./scripts/MiniMaxXlsx.exe reference-check data.xlsx

# Step 2: Examine structure (obtain sheet names, headers)
./scripts/MiniMaxXlsx.exe inspect data.xlsx --pretty

# Step 3: Generate PivotTable (employ --style for theme)
./scripts/MiniMaxXlsx.exe pivot \
    data.xlsx output.xlsx \
    --source "SalesData!A1:F500" \
    --rows "Product,Region" \
    --values "Revenue:sum,Quantity:count" \
    --location "Summary!A3" \
    --name "SalesSummary" \
    --style "finance"

# Step 4: Verify output
./scripts/MiniMaxXlsx.exe validate output.xlsx
# Exit code 0 = safe for delivery
# Exit code 1 = critical issues - MUST RECTIFY
```

</Complete Workflow Example>

<Example Scenarios>

**Sales Summary by Product**:
```bash
./scripts/MiniMaxXlsx.exe pivot \
    sales.xlsx output.xlsx \
    --source "Sales!A1:F500" \
    --rows "Product" \
    --values "Revenue:sum,Units:count" \
    --location "Summary!A3"
```

**Trend Analysis with Line Chart**:
```bash
./scripts/MiniMaxXlsx.exe pivot \
    monthly_data.xlsx trend_report.xlsx \
    --source "Data!A1:D100" \
    --rows "Month" \
    --values "Revenue:sum" \
    --location "Trend!A3" \
    --chart "line"
```

**Market Share with Pie Chart**:
```bash
./scripts/MiniMaxXlsx.exe pivot \
    market_data.xlsx share_report.xlsx \
    --source "Sales!A1:C50" \
    --rows "Region" \
    --values "Sales:sum" \
    --location "Share!A3" \
    --chart "pie"
```

</Example Scenarios>

<When to Use PivotTable>

**Employ PivotTable when**:
- User explicitly requests "pivot table" or "data pivot"
- Task necessitates data summarization by categories
- Dataset contains 50+ rows with grouping requirements
- Cross-tabulation or multi-dimensional analysis required

**Trigger Keywords**: summarize, aggregate, group by, categorize, breakdown, statistics, distribution, count by, total by

</When to Use PivotTable>

<Best Practices>

**Source Data Prerequisites**:
- Initial row MUST contain unique column headers
- No merged cells, no blank rows within data
- Consistent data types in each column

**Location Placement**:
- Employ a NEW sheet for PivotTable (circumvents overwriting data)
- Commence at cell A3 or B2 (leaves space for filter dropdowns)

</Best Practices>

<🚨 ESSENTIAL: DO NOT MODIFY PIVOT OUTPUT WITH OPENPYXL>

**⛔ NEVER open pivot output file using openpyxl following `pivot` command execution!**

openpyxl will corrupt the pivotCache paths upon saving, causing MS Excel to crash.

**When you require Cover sheet or additional styling:**
```
✅ CORRECT WORKFLOW:
1. openpyxl: Generate base.xlsx containing ALL sheets (Cover, Data, etc.)
2. pivot command: base.xlsx → final.xlsx (appends PivotTable as LAST step)
3. validate final.xlsx
4. DELIVER (do NOT modify final.xlsx subsequently)

❌ INCORRECT WORKFLOW (CORRUPTS FILE):
1. pivot command: data.xlsx → pivot.xlsx
2. openpyxl: Open pivot.xlsx, append Cover sheet, save  ← FILE CORRUPTED!
3. MS Excel cannot open the file
```

</🚨 ESSENTIAL: DO NOT MODIFY PIVOT OUTPUT WITH OPENPYXL>

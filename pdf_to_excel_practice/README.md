# PDF to Excel Practice

This folder documents my practice for converting and cleaning text data from PDF files before structured entry into Excel or Google Sheets.

## Task Description
- Read text-based PDF documents
- Identify relevant fields
- Clean extra spaces without changing meaning
- Enter cleaned data into spreadsheet columns
- Flag unclear or unreadable entries for review

## Sample Text Cleaning (Python - Conceptual)

```python
import pandas as pd

data = {
    "Raw_Text": [
        "  Name :  Riya Sharma ",
        " Address :  45   Park  Street ",
        " Phone :  9876543210 "
    ]
}

df = pd.DataFrame(data)
df["Cleaned_Text"] = df["Raw_Text"].str.strip().str.replace("  ", " ", regex=False)
df

import yfinance as yf
import pandas as pd

ticker = yf.Ticker("NVDA")
income_stmt = ticker.financials.T
selected_columns = {
    'Total Revenue': 'Revenue',
    'Cost Of Revenue': 'COGS',
    'Gross Profit': 'Gross Profit',
    'Research And Development': 'R&D',
    'Selling General And Administration': 'SG&A',
    'Operating Income': 'Operating Income',
    'Interest Expense': 'Interest',
    'Pretax Income': 'Pre-tax Income',
    'Tax Provision': 'Taxes',
    'Net Income': 'Net Income'
}

print("Available Columns:\n", income_stmt.columns.tolist())
income_stmt = income_stmt[list(selected_columns.keys())]
income_stmt.rename(columns=selected_columns, inplace=True)
income_stmt.index = income_stmt.index.date
print("NVIDIA Income Statement (Last 4 Years):")
print(income_stmt)
income_stmt.to_excel("NVDA_income_statement.xlsx")

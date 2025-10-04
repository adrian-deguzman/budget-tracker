# --------------------------------------------------------------
# LOCAL TESTING
# --------------------------------------------------------------

# import streamlit as st
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
# from datetime import datetime, date
# import pandas as pd
# import json
# import os
# from dateutil.relativedelta import relativedelta
# from io import BytesIO
# import base64
# from typing import Tuple, Optional, List

# # ---------------------------
# # Config / Constants
# # ---------------------------
# CREDS_FILE = "service_account.json"  # service account JSON filename
# CONFIG_FILE = "config.json"          # local file to store sheet id and options
# DEFAULT_CATEGORIES = [
#     "Bills", "Food Outside", "Shopee", "Meds", "Groceries",
#     "Supplements", "Transpo", "Load", "School", "Savings", "Leisure"
# ]

# SCOPE = ["https://spreadsheets.google.com/feeds",
#          "https://www.googleapis.com/auth/drive"]

# # ---------------------------
# # Utility: config file (local)
# # ---------------------------
# def read_local_config() -> dict:
#     if os.path.exists(CONFIG_FILE):
#         try:
#             with open(CONFIG_FILE, "r") as f:
#                 return json.load(f)
#         except Exception:
#             return {}
#     return {}

# def write_local_config(cfg: dict):
#     with open(CONFIG_FILE, "w") as f:
#         json.dump(cfg, f, indent=2)

# cfg = read_local_config()
# SHEET_ID = cfg.get("SHEET_ID", "")  # can be updated in Settings

# # ---------------------------
# # Google Sheets helpers
# # ---------------------------
# @st.cache_resource
# def get_gspread_client():
#     if not os.path.exists(CREDS_FILE):
#         st.error(f"Service account credentials '{CREDS_FILE}' not found. Put the JSON file in the app folder.")
#         raise FileNotFoundError("Missing service account JSON")
#     creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
#     client = gspread.authorize(creds)
#     return client

# --------------------------------------------------------------
# STREAMLIT DEPLOYMENT
# --------------------------------------------------------------
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date
import pandas as pd
import json
import os
from dateutil.relativedelta import relativedelta
from io import BytesIO
import base64
from typing import Tuple, Optional, List

# ---------------------------
# Config / Constants
# ---------------------------
CREDS_FILE = "service_account.json"  # service account JSON filename (local dev only)
CONFIG_FILE = "config.json"          # local file to store sheet id and options
DEFAULT_CATEGORIES = [
    "Bills", "Food Outside", "Shopee", "Meds", "Groceries",
    "Supplements", "Transpo", "Load", "School", "Savings", "Leisure"
]

SCOPE = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

# ---------------------------
# Utility: config file (local)
# ---------------------------
def read_local_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def write_local_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

cfg = read_local_config()
SHEET_ID = cfg.get("SHEET_ID", "")  # can be updated in Settings

# ---------------------------
# Google Sheets helpers
# ---------------------------
@st.cache_resource
def get_gspread_client():
    creds = None

    # --- 1) Check if running on Streamlit Cloud with st.secrets ---
    if "service_account" in st.secrets:
        service_account_info = dict(st.secrets["service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, SCOPE)

    # --- 2) Otherwise, try local service_account.json ---
    elif os.path.exists(CREDS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)

    else:
        st.error(
            f"Service account credentials not found.\n"
            f"Upload `{CREDS_FILE}` locally OR configure `[service_account]` in Streamlit secrets."
        )
        raise FileNotFoundError("Missing service account JSON")

    client = gspread.authorize(creds)
    return client

# --------------------------------------------------------------

def open_sheet(sheet_id: str):
    client = get_gspread_client()
    try:
        return client.open_by_key(sheet_id)
    except Exception as e:
        st.error("Could not open Google Sheet. Make sure the sheet ID is correct and the service account has editor access.")
        raise

def worksheet_to_df(sh, worksheet_name: str, dtype=None) -> pd.DataFrame:
    try:
        ws = sh.worksheet(worksheet_name)
    except Exception:
        # If worksheet doesn't exist, return empty df
        return pd.DataFrame()
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    if dtype:
        return df.astype(dtype, errors='ignore')
    return df

def append_row_to_sheet(sh, worksheet_name: str, row: List):
    ws = sh.worksheet(worksheet_name)
    ws.append_row(row)

def update_or_append_budget_row(sh, worksheet_name: str, year: int, month: int, update_col: str, value):
    """
    - Looks for a row where year==year and month==month. If found, updates `update_col` (exact header name).
    - If not found, appends a new row with year, month and update_col value.
    """
    ws = sh.worksheet(worksheet_name)
    df = worksheet_to_df(sh, worksheet_name)
    # if sheet empty -> append headers then row
    if df.empty:
        headers = ['year', 'month', update_col]
        ws.clear()
        ws.append_row(headers)
        ws.append_row([int(year), int(month), value])
        return

    # find header index
    headers = df.columns.tolist()
    if 'year' not in headers or 'month' not in headers:
        # rebuild with basic headers and re-append previous rows + new one
        ws.append_row(['year', 'month', update_col])
        ws.append_row([int(year), int(month), value])
        return

    # find rows with matching year & month
    mask = (df['year'].astype(int) == int(year)) & (df['month'].astype(int) == int(month))
    if mask.any():
        # update first matching row's column (if column exists)
        row_idx = df[mask].index[0] + 2  # gspread is 1-indexed and header row
        # determine column index
        if update_col in headers:
            col_idx = headers.index(update_col) + 1
        else:
            # append column at end
            col_idx = len(headers) + 1
            ws.update_cell(1, col_idx, update_col)
        ws.update_cell(row_idx, col_idx, value)
    else:
        # append new row with year, month and value at proper column
        # Create row with proper number of columns
        append = {}
        append['year'] = int(year)
        append['month'] = int(month)
        append[update_col] = value
        # rearrange order as headers in sheet
        new_row = []
        for h in headers:
            new_row.append(append.get(h, ""))
        # if extra columns (update_col) not in headers, we append it then fill
        if update_col not in headers:
            new_row.append(value)
            ws.append_row(new_row)
        else:
            ws.append_row(new_row)

# ---------------------------
# Helper: date ranges and defaults
# ---------------------------
def get_year_options(past_years=5):
    this_year = datetime.now().year
    return [this_year - i for i in range(past_years - 1, -1, -1)]

def get_month_options_for_year(year: int):
    now = datetime.now()
    if year == now.year:
        # months to date (1..current month)
        return list(range(1, now.month + 1))
    else:
        return list(range(1, 13))

def fmt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")

def fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")

# ---------------------------
# Helper: receipts handling (local)
# ---------------------------
RECEIPT_DIR = "receipts"
os.makedirs(RECEIPT_DIR, exist_ok=True)

def save_receipt_file(uploaded_file) -> str:
    """
    Saves uploaded receipt to receipts/ and returns relative path string.
    """
    if uploaded_file is None:
        return ""
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fn = f"{ts}_{uploaded_file.name}"
    path = os.path.join(RECEIPT_DIR, fn)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ---------------------------
# UI Components & Logic (module style)
# ---------------------------
def sidebar_menu():
    st.sidebar.title("SpendWise")
    st.sidebar.markdown("### 📒 Navigation")
    menu = st.sidebar.radio(
        "",
        [
            "Expense Log",
            "Monthly Budget",
            "Budget by Category",
            "Savings Goal",
            "Savings Log",
            "Debt Log",
            "Income Log",
            "Settings"
        ],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Made with ❤️ SpendWise")
    return menu

def load_categories(sh) -> List[str]:
    df = worksheet_to_df(sh, "Categories")
    if df.empty or 'category' not in df.columns:
        # initialize categories sheet (if missing)
        try:
            ws = sh.worksheet("Categories")
        except Exception:
            sh.add_worksheet("Categories", rows=100, cols=2)
            ws = sh.worksheet("Categories")
            ws.append_row(["category"])
            for c in DEFAULT_CATEGORIES:
                ws.append_row([c])
            return DEFAULT_CATEGORIES.copy()
        return DEFAULT_CATEGORIES.copy()
    else:
        cats = df['category'].dropna().astype(str).tolist()
        if not cats:
            return DEFAULT_CATEGORIES.copy()
        return cats

def save_categories(sh, categories: List[str]):
    # overwrite Categories worksheet
    try:
        ws = sh.worksheet("Categories")
        ws.clear()
    except Exception:
        sh.add_worksheet("Categories", rows=200, cols=2)
        ws = sh.worksheet("Categories")
    headers = ["category"]
    ws.append_row(headers)
    for c in categories:
        ws.append_row([c])

def expense_log_page(sh, categories):
    st.header("📥 Expense Log")
    today = date.today()
    with st.form("expense_form", clear_on_submit=False):
        dt_expense = st.date_input("Date of expense", value=today)
        category = st.selectbox("Category", options=categories, index=(categories.index("Food Outside") if "Food Outside" in categories else 0))
        mode = st.selectbox("Mode of payment", options=["Cash", "GCash", "Maribank", "Landbank"])
        amount = st.number_input("Amount (PHP)", min_value=0.0, step=10.0, format="%.2f")
        receipt = st.file_uploader("Receipt (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        submitted = st.form_submit_button("Submit Expense")
        if submitted:
            if amount <= 0:
                st.warning("Please enter an amount greater than 0.")
            else:
                ts = fmt_datetime(datetime.now())
                receipt_path = save_receipt_file(receipt) if receipt else ""
                row = [ts, fmt_date(dt_expense), category, mode, float(amount), receipt_path]
                try:
                    append_row_to_sheet(sh, "Expense Log", row)
                    st.success("Expense recorded ✅")
                    # clear form fields by rerun simple workaround
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to write to Expense Log sheet: {e}")

    # show last 5 expenses with pagination & edit placeholder
    st.subheader("Recent Expenses")
    df = worksheet_to_df(sh, "Expense Log")
    if df.empty:
        st.info("No expenses recorded yet.")
        return
    df_display = df.copy()
    # format amount
    if 'amount' in df_display.columns:
        try:
            df_display['amount'] = df_display['amount'].astype(float).map("₱{:.2f}".format)
        except Exception:
            pass
    # pagination
    page_size = 5
    page = st.session_state.get("expenses_page", 0)
    max_page = (len(df_display) - 1) // page_size
    start = page * page_size
    end = start + page_size
    st.write(df_display.iloc[start:end])
    cols = st.columns(3)
    if cols[0].button("< Prev"):
        st.session_state["expenses_page"] = max(0, page - 1)
        st.rerun()
    st.write("")
    if cols[2].button("Next >"):
        st.session_state["expenses_page"] = min(max_page, page + 1)
        st.rerun()

def monthly_budget_page(sh, categories):
    st.header("💰 Monthly Budget")
    years = get_year_options(6)
    default_year = datetime.now().year
    col1, col2 = st.columns(2)
    with st.form("monthly_budget_form"):
        year = st.selectbox("Year", years, index=years.index(default_year))
        month_opts = get_month_options_for_year(year)
        default_month = datetime.now().month
        month = st.selectbox("Month", month_opts, index=(month_opts.index(default_month) if default_month in month_opts else 0))
        # default total budget = previous month total if exists
        prev_month = date(year, month, 1) - relativedelta(months=1)
        prev_df = worksheet_to_df(sh, "Monthly Budget")
        default_budget = ""
        if not prev_df.empty:
            m = prev_df[(prev_df['year'].astype(int) == prev_month.year) & (prev_df['month'].astype(int) == prev_month.month)]
            if not m.empty:
                # try first available numeric column for budget
                try:
                    # if there's a 'total budget' column use that
                    if 'total budget' in m.columns:
                        default_budget = float(m['total budget'].iloc[-1]) if m['total budget'].iloc[-1] != "" else ""
                    else:
                        # pick numeric-like column values ignoring year & month
                        for c in m.columns:
                            if c not in ['year', 'month'] and str(m[c].iloc[-1]).strip() != "":
                                try:
                                    default_budget = float(m[c].iloc[-1])
                                    break
                                except:
                                    continue
                except Exception:
                    default_budget = ""
        total_budget = st.number_input("Total budget (PHP)", value=(float(default_budget) if default_budget != "" else 0.0), format="%.2f")
        submit = st.form_submit_button("Submit Monthly Budget")
        if submit:
            # update Monthly Budget sheet 'total budget' column for the year/month
            try:
                # ensure Monthly Budget sheet exists
                try:
                    sh.worksheet("Monthly Budget")
                except Exception:
                    sh.add_worksheet("Monthly Budget", rows=200, cols=10)
                    sh.worksheet("Monthly Budget").append_row(['year','month','total budget'])
                update_or_append_budget_row(sh, "Monthly Budget", int(year), int(month), "total budget", float(total_budget))
                st.success("Monthly budget updated ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update Monthly Budget: {e}")

    # display entered monthly budget and sum of category budgets for given year & month
    mb_df = worksheet_to_df(sh, "Monthly Budget")
    bycat_df = worksheet_to_df(sh, "Budget by Category")
    entered = 0.0
    if not mb_df.empty and 'year' in mb_df.columns and 'month' in mb_df.columns:
        filt = (mb_df['year'].astype(int) == int(year)) & (mb_df['month'].astype(int) == int(month))
        if filt.any():
            # pick 'total budget' if present else take first numeric column besides year/month
            try:
                if 'total budget' in mb_df.columns:
                    v = mb_df.loc[filt, 'total budget'].dropna().astype(float)
                    if not v.empty:
                        entered = float(v.iloc[-1])
                else:
                    for c in mb_df.columns:
                        if c not in ['year', 'month']:
                            try:
                                v = mb_df.loc[filt, c].dropna().astype(float)
                                if not v.empty:
                                    entered = float(v.iloc[-1]); break
                            except:
                                continue
            except Exception:
                entered = 0.0

    # sum category budgets
    sum_cat = 0.0
    if not bycat_df.empty and all(x in bycat_df.columns for x in ['year','month','budget']):
        filt2 = (bycat_df['year'].astype(int) == int(year)) & (bycat_df['month'].astype(int) == int(month))
        if filt2.any():
            try:
                sum_cat = bycat_df.loc[filt2, 'budget'].astype(float).sum()
            except Exception:
                sum_cat = 0.0

    st.markdown("---")
    st.markdown(f"**Entered Monthly Budget:** ₱{entered:,.2f}")
    st.markdown(f"**Sum of Category Budgets:** ₱{sum_cat:,.2f}")
    diff = entered - sum_cat
    if diff >= 0:
        st.success(f"You are ₱{diff:,.2f} under your monthly budget.")
    else:
        st.warning(f"You are ₱{abs(diff):,.2f} over your monthly budget.")

def budget_by_category_page(sh, categories):
    st.header("📊 Budget by Category")
    years = get_year_options(6)
    default_year = datetime.now().year
    with st.form("budget_by_category_form"):
        year = st.selectbox("Year", years, index=years.index(default_year))
        month_opts = get_month_options_for_year(year)
        default_month = datetime.now().month
        month = st.selectbox("Month", month_opts, index=(month_opts.index(default_month) if default_month in month_opts else 0))
        st.write("Enter budgets for categories below. Leave empty to keep previous value.")
        # show each category with default previous month value
        prev_month = date(year, month, 1) - relativedelta(months=1)
        b_df = worksheet_to_df(sh, "Budget by Category")
        cat_values = {}
        for c in categories:
            default_val = 0.0
            if not b_df.empty and all(x in b_df.columns for x in ['year','month','category','budget']):
                prev = b_df[
                    (b_df['year'].astype(int) == prev_month.year) &
                    (b_df['month'].astype(int) == prev_month.month) &
                    (b_df['category'].astype(str) == c)
                ]
                if not prev.empty:
                    try:
                        default_val = float(prev['budget'].iloc[-1])
                    except:
                        default_val = 0.0
            cat_values[c] = st.number_input(f"{c} budget (PHP)", value=float(default_val), format="%.2f", key=f"cat_{c}")
        remarks = st.text_area("Remarks (optional)")
        submit = st.form_submit_button("Submit Category Budgets")
        if submit:
            # for each category append/update Budget by Category sheet
            try:
                try:
                    sh.worksheet("Budget by Category")
                except Exception:
                    sh.add_worksheet("Budget by Category", rows=1000, cols=10)
                    sh.worksheet("Budget by Category").append_row(['year','month','category','budget','remarks'])
                for c, val in cat_values.items():
                    # Either update existing row for year,month,category or append
                    # Simpler: append a new row (Apps Script will manage duplicates if desired),
                    # but to follow spec we try to update existing row for that combination:
                    df = worksheet_to_df(sh, "Budget by Category")
                    if not df.empty and all(x in df.columns for x in ['year','month','category']):
                        mask = (df['year'].astype(int) == int(year)) & (df['month'].astype(int) == int(month)) & (df['category'].astype(str) == c)
                        if mask.any():
                            # find row index and update budget + remarks
                            row_idx = df[mask].index[0] + 2
                            # columns may vary, find appropriate column index
                            headers = df.columns.tolist()
                            if 'budget' in headers:
                                budget_col = headers.index('budget') + 1
                                sh.worksheet("Budget by Category").update_cell(row_idx, budget_col, float(val))
                            else:
                                sh.worksheet("Budget by Category").update_cell(1, len(headers)+1, 'budget')
                                sh.worksheet("Budget by Category").update_cell(row_idx, len(headers)+1, float(val))
                            if remarks:
                                if 'remarks' in headers:
                                    remarks_col = headers.index('remarks') + 1
                                else:
                                    remarks_col = len(headers)+1
                                    sh.worksheet("Budget by Category").update_cell(1, remarks_col, 'remarks')
                                sh.worksheet("Budget by Category").update_cell(row_idx, remarks_col, remarks)
                        else:
                            sh.worksheet("Budget by Category").append_row([int(year), int(month), c, float(val), remarks or ""])
                    else:
                        sh.worksheet("Budget by Category").append_row([int(year), int(month), c, float(val), remarks or ""])
                st.success("Category budgets updated ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update Budget by Category: {e}")

def savings_goal_page(sh):
    st.header("🏦 Savings Goal")
    years = get_year_options(6)
    default_year = datetime.now().year
    with st.form("savings_goal_form"):
        year = st.selectbox("Year", years, index=years.index(default_year))
        month_opts = get_month_options_for_year(year)
        default_month = datetime.now().month
        month = st.selectbox("Month", month_opts, index=(month_opts.index(default_month) if default_month in month_opts else 0))
        # default amount from previous month
        prev_month = date(year, month, 1) - relativedelta(months=1)
        sg_df = worksheet_to_df(sh, "Savings Goal")
        default_amt = 0.0
        if not sg_df.empty and all(x in sg_df.columns for x in ['year','month','amount']):
            prev = sg_df[(sg_df['year'].astype(int)==prev_month.year)&(sg_df['month'].astype(int)==prev_month.month)]
            if not prev.empty:
                try:
                    default_amt = float(prev['amount'].iloc[-1])
                except:
                    default_amt = 0.0
        amount = st.number_input("Savings Goal amount (PHP)", value=float(default_amt), format="%.2f")
        purpose = st.text_input("Purpose (optional)")
        submit = st.form_submit_button("Submit Savings Goal")
        if submit:
            try:
                try:
                    sh.worksheet("Savings Goal")
                except:
                    sh.add_worksheet("Savings Goal", rows=500, cols=10)
                    sh.worksheet("Savings Goal").append_row(['year','month','amount','purpose'])
                update_or_append_budget_row(sh, "Savings Goal", int(year), int(month), "amount", float(amount))
                # update purpose as well (append/update purpose column)
                df = worksheet_to_df(sh, "Savings Goal")
                if not df.empty:
                    mask = (df['year'].astype(int)==int(year)) & (df['month'].astype(int)==int(month))
                    if mask.any():
                        row_idx = df[mask].index[0] + 2
                        headers = df.columns.tolist()
                        if 'purpose' in headers:
                            purpose_col = headers.index('purpose') + 1
                        else:
                            purpose_col = len(headers) + 1
                            sh.worksheet("Savings Goal").update_cell(1, purpose_col, 'purpose')
                        sh.worksheet("Savings Goal").update_cell(row_idx, purpose_col, purpose or "")
                st.success("Savings goal updated ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update Savings Goal: {e}")

    # Display savings goal for selected month and current savings (sum of Savings Log)
    sg_df = worksheet_to_df(sh, "Savings Goal")
    sg_amt = 0.0
    if not sg_df.empty and all(x in sg_df.columns for x in ['year','month','amount']):
        sel = (sg_df['year'].astype(int)==int(year)) & (sg_df['month'].astype(int)==int(month))
        if sel.any():
            try:
                sg_amt = float(sg_df.loc[sel, 'amount'].iloc[-1])
            except:
                sg_amt = 0.0

    # Current Savings: sum of Savings Log for that month/year
    s_log = worksheet_to_df(sh, "Savings Log")
    curr_savings = 0.0
    if not s_log.empty and all(x in s_log.columns for x in ['date','amount']):
        # filter by year & month
        s_log['date'] = pd.to_datetime(s_log['date'], errors='coerce')
        filt = (s_log['date'].dt.year == int(year)) & (s_log['date'].dt.month == int(month))
        if filt.any():
            try:
                curr_savings = s_log.loc[filt, 'amount'].astype(float).sum()
            except:
                curr_savings = 0.0

    st.markdown("---")
    st.markdown(f"**Savings Goal for {datetime(year, month, 1).strftime('%B')}:** ₱{sg_amt:,.2f}")
    st.markdown(f"**Current Savings:** ₱{curr_savings:,.2f}")
    diff = sg_amt - curr_savings
    if diff > 0:
        st.info(f"You are ₱{diff:,.2f} under your monthly savings goal.")
    else:
        st.success("You've reached or exceeded your monthly savings goal 🎉")

def savings_log_page(sh):
    st.header("💾 Savings Log")
    today = date.today()
    with st.form("savings_log_form"):
        dt_s = st.date_input("Date", value=today)
        source = st.text_input("Source")
        add_or_reduce = st.selectbox("Action", ["Add savings", "Reduce savings"])
        amount = st.number_input("Amount (PHP)", min_value=0.0, step=10.0, format="%.2f")
        submit = st.form_submit_button("Submit Savings Log")
        if submit:
            if amount <= 0:
                st.warning("Enter amount greater than 0.")
            else:
                recorded = float(amount) if add_or_reduce == "Add savings" else -float(amount)
                ts = fmt_datetime(datetime.now())
                row = [ts, fmt_date(dt_s), source, recorded]
                try:
                    # ensure sheet exists
                    try:
                        sh.worksheet("Savings Log")
                    except:
                        sh.add_worksheet("Savings Log", rows=1000, cols=10)
                        sh.worksheet("Savings Log").append_row(['datetime','date','source','amount'])
                    append_row_to_sheet(sh, "Savings Log", row)
                    st.success("Savings log updated ✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to append to Savings Log: {e}")

def debt_log_page(sh):
    st.header("💳 Debt Log")
    # Compute current debts per person from Debt Log sheet
    debt_df = worksheet_to_df(sh, "Debt Log")
    if debt_df.empty or not all(x in debt_df.columns for x in ['date','person','amount']):
        st.info("No debts recorded yet.")
        current_debts = {}
    else:
        # Sum amounts per person (assuming positive = you owe or others owe you depending on convention)
        try:
            debt_df['date'] = pd.to_datetime(debt_df['date'], errors='coerce')
            sum_per_person = debt_df.groupby('person')['amount'].apply(lambda s: s.astype(float).sum())
            current_debts = sum_per_person.to_dict()
        except Exception:
            current_debts = {}

    st.subheader("Current debts")
    if current_debts:
        for person, amt in current_debts.items():
            st.write(f"**{person}** — ₱{float(amt):,.2f}")
    else:
        st.write("No current debts to display.")

    st.markdown("---")
    # New or Old debt logic
    with st.form("debt_form"):
        new_or_old = st.selectbox("New or existing debt", ["New"] + list(sorted(current_debts.keys())) )
        dt_d = st.date_input("Date", value=date.today())
        if new_or_old == "New":
            person = st.text_input("Person name")
            amount = st.number_input("Amount (PHP)", min_value=0.0, step=10.0, format="%.2f")
            update_action = "add"  # not applicable
            update_amount = amount
        else:
            person = new_or_old
            # display current amount (non-editable) and allow add or reduce
            current_amt = float(current_debts.get(person, 0.0))
            st.markdown(f"Current amount for **{person}**: ₱{current_amt:,.2f}")
            update_action = st.selectbox("Update with", ["Add debt", "Reduce debt"])
            update_amount = st.number_input("Update amount (PHP)", min_value=0.0, step=10.0, format="%.2f")
            if update_amount < 0:
                update_amount = abs(update_amount)
        submit = st.form_submit_button("Submit Debt Update")
        if submit:
            if not person:
                st.warning("Please provide a person name.")
            else:
                ts = fmt_datetime(datetime.now())
                if new_or_old == "New":
                    row = [ts, fmt_date(dt_d), person, float(amount)]
                else:
                    if update_action == "Add debt":
                        new_recorded = current_amt + float(update_amount)
                    else:
                        new_recorded = current_amt - float(update_amount)
                    row = [ts, fmt_date(dt_d), person, float(new_recorded)]
                try:
                    try:
                        sh.worksheet("Debt Log")
                    except:
                        sh.add_worksheet("Debt Log", rows=1000, cols=10)
                        sh.worksheet("Debt Log").append_row(['datetime','date','person','amount'])
                    append_row_to_sheet(sh, "Debt Log", row)
                    st.success("Debt log updated ✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update Debt Log: {e}")

def income_log_page(sh):
    st.header("💵 Income Log")
    today = date.today()
    with st.form("income_form"):
        dt_i = st.date_input("Date", value=today)
        source = st.text_input("Source")
        amount = st.number_input("Amount (PHP)", min_value=0.0, step=50.0, format="%.2f")
        submit = st.form_submit_button("Submit Income")
        if submit:
            if amount <= 0:
                st.warning("Enter amount > 0.")
            else:
                ts = fmt_datetime(datetime.now())
                row = [ts, fmt_date(dt_i), source, float(amount)]
                try:
                    try:
                        sh.worksheet("Income Log")
                    except:
                        sh.add_worksheet("Income Log", rows=1000, cols=10)
                        sh.worksheet("Income Log").append_row(['datetime','date','source','amount'])
                    append_row_to_sheet(sh, "Income Log", row)
                    st.success("Income recorded ✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to append to Income Log: {e}")

# def settings_page(sh):
#     st.header("⚙️ Settings")
#     st.info("Set the master Google Sheet ID and manage categories.")
#     # sheet id input
#     global SHEET_ID
#     st.text("Google Sheet ID")
#     sid = st.text_input("Sheet ID (open your Google Sheet and copy the long ID in the URL)", value=cfg.get("SHEET_ID", ""))
#     if st.button("Save Sheet ID"):
#         cfg_local = read_local_config()
#         cfg_local['SHEET_ID'] = sid.strip()
#         write_local_config(cfg_local)
#         st.success("Saved Sheet ID. Please reload the app (browser refresh).")
#     st.markdown("---")
#     # categories editing
#     st.subheader("Categories")
#     cats = load_categories(sh)
#     st.write("Current categories (edit or re-order):")
#     for i, c in enumerate(cats):
#         new_name = st.text_input(f"cat_{i}", value=c, key=f"cat_edit_{i}")
#         cats[i] = new_name
#         if st.button(f"Delete {c}", key=f"del_{i}"):
#             cats.pop(i)
#             save_categories(sh, cats)
#             st.success("Deleted category")
#             st.rerun()
#     st.markdown("Add new category")
#     new_cat = st.text_input("New category name")
#     if st.button("Add Category"):
#         if new_cat.strip() != "":
#             cats.append(new_cat.strip())
#             save_categories(sh, cats)
#             st.success("Category added")
#             st.rerun()

#     if st.button("Save Categories (overwrite)"):
#         save_categories(sh, cats)
#         st.success("Categories saved to sheet.")

#     st.markdown("---")
#     st.subheader("App preferences")
#     st.write("Local receipt storage folder:", RECEIPT_DIR)
#     if st.button("Clear local receipts"):
#         for f in os.listdir(RECEIPT_DIR):
#             try:
#                 os.remove(os.path.join(RECEIPT_DIR, f))
#             except:
#                 pass
#         st.success("Cleared receipts folder.")

# def settings_page(sh):
#     st.header("⚙️ Settings")
#     st.info("Set the master Google Sheet ID and manage categories.")
#     # sheet id input
#     global SHEET_ID
#     st.text("Google Sheet ID")
#     sid = st.text_input("Sheet ID (open your Google Sheet and copy the long ID in the URL)", value=cfg.get("SHEET_ID", ""))
#     if st.button("Save Sheet ID"):
#         cfg_local = read_local_config()
#         cfg_local['SHEET_ID'] = sid.strip()
#         write_local_config(cfg_local)
#         st.success("Saved Sheet ID. Please reload the app (browser refresh).")

#     st.markdown("---")
#     # categories editing (customization structure)
#     st.subheader("Categories")
#     cats = load_categories(sh)

#     if "edit_category_idx" not in st.session_state:
#         st.session_state["edit_category_idx"] = None

#     for i, c in enumerate(cats):
#         cols = st.columns([6,1,1])
#         with cols[0]:
#             if st.session_state["edit_category_idx"] == i:
#                 new_name = st.text_input("Edit category", value=c, key=f"edit_cat_{i}")
#             else:
#                 st.text(c)
#         with cols[1]:
#             if st.session_state["edit_category_idx"] == i:
#                 if st.button("💾", key=f"save_cat_{i}"):
#                     cats[i] = new_name.strip()
#                     save_categories(sh, cats)
#                     st.session_state["edit_category_idx"] = None
#                     st.success("Category updated")
#                     st.rerun()
#             else:
#                 if st.button("✏️", key=f"edit_cat_{i}"):
#                     st.session_state["edit_category_idx"] = i
#                     st.rerun()
#         with cols[2]:
#             if st.button("🗑️", key=f"del_cat_{i}"):
#                 cats.pop(i)
#                 save_categories(sh, cats)
#                 st.success("Deleted category")
#                 st.rerun()

#     st.markdown("Add new category")
#     new_cat = st.text_input("New category name")
#     if st.button("Add Category"):
#         if new_cat.strip() != "":
#             cats.append(new_cat.strip())
#             save_categories(sh, cats)
#             st.success("Category added")
#             st.rerun()

#     if st.button("Save Categories (overwrite)"):
#         save_categories(sh, cats)
#         st.success("Categories saved to sheet.")

#     st.markdown("---")
#     st.subheader("App preferences")
#     st.write("Local receipt storage folder:", RECEIPT_DIR)
#     if st.button("Clear local receipts"):
#         for f in os.listdir(RECEIPT_DIR):
#             try:
#                 os.remove(os.path.join(RECEIPT_DIR, f))
#             except:
#                 pass
#         st.success("Cleared receipts folder.")

def settings_page(sh):
    st.header("⚙️ Settings")
    st.info("Set the master Google Sheet ID and manage categories.")
    # sheet id input
    global SHEET_ID
    st.text("Google Sheet ID")
    sid = st.text_input("Sheet ID (open your Google Sheet and copy the long ID in the URL)", value=cfg.get("SHEET_ID", ""))
    if st.button("Save Sheet ID"):
        cfg_local = read_local_config()
        cfg_local['SHEET_ID'] = sid.strip()
        write_local_config(cfg_local)
        st.success("Saved Sheet ID. Please reload the app (browser refresh).")

    st.markdown("---")
    # categories editing (customization structure)
    st.subheader("Categories")
    cats = load_categories(sh)

    if "edit_category_idx" not in st.session_state:
        st.session_state["edit_category_idx"] = None

    for i, c in enumerate(cats):
        cols = st.columns([6,1,1])
        with cols[0]:
            if st.session_state["edit_category_idx"] == i:
                new_name = st.text_input("Edit category", value=c, key=f"edit_cat_{i}")
            else:
                st.text(c)
        with cols[1]:
            if st.session_state["edit_category_idx"] == i:
                if st.button("Save", key=f"save_cat_{i}"):
                    cats[i] = new_name.strip()
                    save_categories(sh, cats)
                    st.session_state["edit_category_idx"] = None
                    st.success("Category updated")
                    st.rerun()
            else:
                if st.button("Edit", key=f"edit_cat_{i}"):
                    st.session_state["edit_category_idx"] = i
                    st.rerun()
        with cols[2]:
            if st.button("Delete", key=f"del_cat_{i}"):
                cats.pop(i)
                save_categories(sh, cats)
                st.success("Deleted category")
                st.rerun()

    st.markdown("Add new category")
    new_cat = st.text_input("New category name")
    if st.button("Add Category"):
        if new_cat.strip() != "":
            cats.append(new_cat.strip())
            save_categories(sh, cats)
            st.success("Category added")
            st.rerun()

    if st.button("Save Categories (overwrite)"):
        save_categories(sh, cats)
        st.success("Categories saved to sheet.")

    st.markdown("---")
    st.subheader("App preferences")
    st.write("Local receipt storage folder:", RECEIPT_DIR)
    if st.button("Clear local receipts"):
        for f in os.listdir(RECEIPT_DIR):
            try:
                os.remove(os.path.join(RECEIPT_DIR, f))
            except:
                pass
        st.success("Cleared receipts folder.")




# ---------------------------
# App main
# ---------------------------
def main():
    st.set_page_config(page_title="SpendWise", layout="wide")
    st.title("SpendWise — Personal Finance Helper")

    # show helpful note about service account
    st.sidebar.markdown("**Setup tip:** Make sure your service account email (from `service_account.json`) is added as Editor to the Google Sheet.")
    # top-level get sheet id from config
    cfg_local = read_local_config()
    sheet_id = cfg_local.get("SHEET_ID", "")
    if not sheet_id:
        st.sidebar.warning("No Google Sheet ID configured. Go to Settings to add your master sheet ID.")
    menu = sidebar_menu()

    # If sheet id configured, open sheet and get categories
    sh = None
    categories = DEFAULT_CATEGORIES.copy()
    if sheet_id:
        try:
            sh = open_sheet(sheet_id)
            categories = load_categories(sh)
        except Exception as e:
            st.sidebar.error("Could not access Google Sheet. Check config and service account.")
            # still show settings page
            if menu != "Settings":
                st.info("Switch to Settings to set or update the Sheet ID.")
    else:
        # create a lightweight demo mode (no write)
        st.sidebar.info("Running in limited mode (no sheet). Add a Sheet ID in Settings to enable full functionality.")

    # dispatch pages (pass sh even if None)
    try:
        if menu == "Expense Log":
            if not sh:
                st.warning("Connect your Google Sheet in Settings to use this page.")
            else:
                expense_log_page(sh, categories)
        elif menu == "Monthly Budget":
            if not sh:
                st.warning("Connect your Google Sheet in Settings to use this page.")
            else:
                monthly_budget_page(sh, categories)
        elif menu == "Budget by Category":
            if not sh:
                st.warning("Connect your Google Sheet in Settings to use this page.")
            else:
                budget_by_category_page(sh, categories)
        elif menu == "Savings Goal":
            if not sh:
                st.warning("Connect your Google Sheet in Settings to use this page.")
            else:
                savings_goal_page(sh)
        elif menu == "Savings Log":
            if not sh:
                st.warning("Connect your Google Sheet in Settings to use this page.")
            else:
                savings_log_page(sh)
        elif menu == "Debt Log":
            if not sh:
                st.warning("Connect your Google Sheet in Settings to use this page.")
            else:
                debt_log_page(sh)
        elif menu == "Income Log":
            if not sh:
                st.warning("Connect your Google Sheet in Settings to use this page.")
            else:
                income_log_page(sh)
        elif menu == "Settings":
            # If user hasn't set sheet id in config, allow editing a sheet id field here
            if not sh:
                st.info("Enter your Google Sheet ID below to connect the app to your master file.")
                sid = st.text_input("Master Google Sheet ID", value=sheet_id)
                if st.button("Save and connect"):
                    cfg_local['SHEET_ID'] = sid.strip()
                    write_local_config(cfg_local)
                    st.success("Saved Sheet ID. Please refresh the page to connect.")
            else:
                settings_page(sh)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

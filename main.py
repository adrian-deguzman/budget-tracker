import streamlit as st
from streamlit_option_menu import option_menu
import calculator
import budgettracker

st.set_page_config(page_title="SpendWise Tools", page_icon="💼", layout="centered")

# ---- TOP NAVBAR ----
selected = option_menu(
    menu_title=None,
    options=["Home", "Calculator", "Budget Tracker"],
    icons=["house", "calculator", "wallet2"],
    orientation="horizontal",
    default_index=0,
)

# ---- PAGE ROUTING ----
if selected == "Home":
    st.title("💼 SpendWise Tools")
    st.write("""
    Welcome to **SpendWise**, a simple multi-tool app built with Streamlit.  
    Use the top menu to switch between tools:
    - 🧮 Calculator  
    - 💰 Budget Tracker
    """)
    st.info("Select a feature from the top bar to start!")
elif selected == "Calculator":
    calculator.run()
elif selected == "Budget Tracker":
    budgettracker.run()

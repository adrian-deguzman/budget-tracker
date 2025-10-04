import streamlit as st

def run():
    st.title("💰 Simple Budget Tracker")

    if "budget" not in st.session_state:
        st.session_state.budget = 0.0
    if "expenses" not in st.session_state:
        st.session_state.expenses = []

    st.subheader("Set Your Total Budget")
    budget = st.number_input("Enter your budget amount (₱):", min_value=0.0, step=100.0)
    if st.button("Set Budget"):
        st.session_state.budget = budget
        st.success(f"Budget set to ₱{budget:,.2f}")

    st.divider()

    st.subheader("Add an Expense")
    desc = st.text_input("Expense description:")
    amount = st.number_input("Expense amount (₱):", min_value=0.0, step=10.0)
    if st.button("Add Expense"):
        if desc and amount > 0:
            st.session_state.expenses.append({"desc": desc, "amount": amount})
            st.success(f"Added '{desc}' - ₱{amount:,.2f}")
        else:
            st.warning("Please enter valid details.")

    st.divider()

    st.subheader("Summary")
    total_spent = sum(exp["amount"] for exp in st.session_state.expenses)
    remaining = st.session_state.budget - total_spent

    col1, col2, col3 = st.columns(3)
    col1.metric("💸 Total Budget", f"₱{st.session_state.budget:,.2f}")
    col2.metric("🧾 Total Spent", f"₱{total_spent:,.2f}")
    col3.metric("💰 Remaining", f"₱{remaining:,.2f}")

    if st.session_state.expenses:
        st.subheader("Expense List")
        for i, exp in enumerate(st.session_state.expenses, start=1):
            st.write(f"{i}. {exp['desc']} - ₱{exp['amount']:,.2f}")
    else:
        st.info("No expenses yet.")

    st.caption("Note: Data resets when you refresh (no database used).")

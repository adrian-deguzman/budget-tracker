import streamlit as st

def run():
    st.title("🧮 Simple Calculator")

    num1 = st.number_input("Enter first number:")
    num2 = st.number_input("Enter second number:")
    operation = st.selectbox("Choose operation:", ["Add", "Subtract", "Multiply", "Divide"])

    if st.button("Calculate"):
        if operation == "Add":
            result = num1 + num2
        elif operation == "Subtract":
            result = num1 - num2
        elif operation == "Multiply":
            result = num1 * num2
        elif operation == "Divide":
            result = num1 / num2 if num2 != 0 else "Cannot divide by zero"
        st.success(f"Result: {result}")

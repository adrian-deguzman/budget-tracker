import streamlit as st

st.title("Simple Calculator")

# Input numbers
num1 = st.number_input("Enter first number", value=0.0)
num2 = st.number_input("Enter second number", value=0.0)

# Operation
operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"])

# Compute
if st.button("Calculate"):
    if operation == "Add":
        result = num1 + num2
    elif operation == "Subtract":
        result = num1 - num2
    elif operation == "Multiply":
        result = num1 * num2
    elif operation == "Divide":
        result = "Error: Division by zero" if num2 == 0 else num1 / num2

    st.success(f"Result: {result}")

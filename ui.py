import os
import pandas as pd
import streamlit as st

# Function to load CSV files from a folder
def load_data(folder):
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data = {}
    for file in csv_files:
        filepath = os.path.join(folder, file)
        data[file] = pd.read_csv(filepath)
    return data

# Streamlit app function
def main():
    # Sidebar input to specify the folder
    folder = st.sidebar.text_input("Enter the path to the folder containing CSV files:")

    if folder and os.path.exists(folder):
        # Load data
        data = load_data(folder)
        if not data:
            st.error("No CSV files found in the specified folder.")
            return
        
        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Previous Row"):
                st.session_state.current_row -= 1
        with col3:
            if st.button("Next Row ➡️"):
                st.session_state.current_row += 1

        # Dropdown for selecting attack (file)
        attack_name = st.sidebar.selectbox("Select an attack", list(data.keys()))
        df = data[attack_name]

        # Initialize the current row index in session state
        if "current_row" not in st.session_state:
            st.session_state.current_row = 0

        # Ensure the current row doesn't exceed dataset bounds
        if st.session_state.current_row < 0:
            st.session_state.current_row = len(df)-1
        elif st.session_state.current_row >= len(df):
            st.session_state.current_row = 0

        # Display current row
        current_row = st.session_state.current_row
        row_data = df.iloc[current_row]

        st.header(f"Attack: {attack_name}")
        st.subheader(f"Row {current_row + 1} of {len(df)}")

        # Display sentence data
        st.markdown(f"**Original:** {row_data['original']}")
        st.markdown(f"**Perturbed:** {row_data['perturbed']}")

        # Display metrics with type consistency
        metrics = {
            "True Label": str(row_data['True']),
            "Prediction (Original)": str(row_data['Pred_original']),
            "Prediction (Perturbed)": str(row_data['Pred_perturbed']),
            "Success": str(row_data['success']),
            "Char-Level Distance": str(row_data['Dist_char']),
            "Token-Level Distance": str(row_data['Dist_token']),
            "Similarity": str(row_data['similarity']),
            "Time": str(row_data['time']),
        }
        st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

    else:
        st.info("Please enter a valid folder path to load CSV files.")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Adversarial Attack Explorer")
    st.title("Adversarial Attack Explorer")
    main()

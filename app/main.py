# Streamlit UI Dashboard (Milestone 5)
import streamlit as st

def main():
    st.set_page_config(
        page_title="Semantic Document Explorer",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Semantic Document Explorer & Q&A")
    st.write("Welcome to the Semantic Document Explorer setup. Day 1 setup is complete!")
    
    st.sidebar.header("Document Control Panel")
    st.sidebar.info("Upload documents and adjust thresholds here in later phases.")

if __name__ == "__main__":
    main()

import streamlit as st
import sys
sys.path.append("C:/Users/carlo/Desktop/Github_repos/audio-analyser")

# import app.pages.signal_visualizer_page as signal_visualizer_page

def main():
    def show_home_page():
        st.title("Audio Analyser")


    home_page = st.Page(show_home_page, title="Home")
    signal_visualizer_page = st.Page("signal_visualizer_page.py", title="Signal Visualizer")
    frequency_extractor_page = st.Page("frequency_extractor_page.py", title="Frequency Extractor")

    pg = st.navigation([home_page, signal_visualizer_page, frequency_extractor_page])
    pg.run()

if __name__ == "__main__":
    main()
import streamlit as st
import sys
sys.path.append("C:/Users/carlo/Desktop/Github_repos/audio-analyser")

def main():
    def wide_space_default():
        st.set_page_config(layout="wide")

    wide_space_default()
    
    def show_home_page():
        st.title("Audio Analyser")
        st.markdown(f"""
                    This is a small web application to help visualize the intuition behind the definition and computation of the Discrete Fourier Transform of a signal.
                    - Go to **DFT Visualizer** to play around with some visualizations
                    - Go to **DFT Frequency Extractor** to actually apply the DFT on an unploaded wav file
                    """
        )


    home_page = st.Page(show_home_page, title="Home")
    dft_visualizer_page = st.Page("dft_visualizer_page.py", title="DFT Visualizer")
    dft_frequency_extractor_page = st.Page("dft_frequency_extractor_page.py", title="DFT Frequency Extractor")

    pg = st.navigation([home_page, dft_visualizer_page, dft_frequency_extractor_page])
    pg.run()

if __name__ == "__main__":
    main()
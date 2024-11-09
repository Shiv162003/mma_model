import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from streamlit_option_menu import option_menu

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compare_audio_files(audio_file_1, audio_file_2):
    def load_audio(file):
        # Load audio file
        y, sr = librosa.load(file, sr=None)
        return y, sr

    def plot_melspectrogram(y, sr, title):
        # Generate Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()

    # Streamlit layout
    st.title("Audio File Comparison")

    # Split the page into two columns for side-by-side comparison
    col1, col2 = st.columns(2)

    # Audio File 1 (Normal Audio)
    with col1:
        st.header("Normal Audio (No Scream)")
        y1, sr1 = load_audio(audio_file_1)
        plot_melspectrogram(y1, sr1, "Mel Spectrogram - Normal Audio")
        st.pyplot(plt)
        st.audio(audio_file_1, format='audio/wav')

    # Audio File 2 (Audio with Scream)
    with col2:
        st.header("Audio with Scream")
        y2, sr2 = load_audio(audio_file_2)
        plot_melspectrogram(y2, sr2, "Mel Spectrogram - Audio with Scream")
        st.pyplot(plt)
        st.audio(audio_file_2, format='audio/wav')


def main():
    st.set_page_config(
        page_title="Multimodal Hate Speech Detector", 
        page_icon="assets/logo.jpeg", 
        layout="wide"
    )
    
    # Set up the option menu for navigation
    selected = option_menu(
        menu_title=None,
        options=["Home", "About Text Modality", "About Audio Modality", "Fusion Techniques & Results"],
        orientation="horizontal",
    )
    
    # Home page
    if selected == "Home":
        st.markdown("""<h1 style='text-align: center;'>Welcome to the Multimodal Hate Speech Detector</h1>""", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("What is Hate Speech?")
            st.markdown("""
                Hate speech is any form of communication that incites violence, hatred, or discrimination 
                against individuals or groups based on attributes such as race, religion, gender, or 
                other characteristics. Detecting hate speech is critical to creating safe online environments 
                and protecting users from harmful content.
            """)
            
            st.header("How This Multimodal Approach Works")
            st.markdown("""
                Our system uses a **multimodal approach** that combines audio and text data to accurately 
                classify hate speech. By using both **melspectrograms** for audio data and **text processing** 
                techniques, we leverage multiple data sources for robust and nuanced detection. 
                Here’s how each component contributes:
                - **Melspectrograms** capture audio features and tone, which are essential for understanding 
                  spoken hate speech.
                - **Text analysis** helps interpret the content and context, improving detection accuracy.
            """)
        with col2:
            st.image('assets/image_2.jpg', width=900, use_container_width=True)

        st.markdown("""<h1 style='text-align: center;'>About the Multimodal Hate Speech Detection Project</h1>""", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1]) 
        with col2:
            st.header("Project Overview")
            st.markdown("""
                **Multimodal Hate Speech Detector** combines advanced natural language processing (NLP) 
                and audio processing to detect hate speech across various forms of communication. 
                By analyzing text and audio features together, our system can identify hate speech with 
                higher accuracy, considering tone, language, and context. This innovative approach offers 
                insights into online behavior patterns, helping users and platforms tackle hate speech 
                effectively.
            """)
            st.header("How the Detection System Works")
            st.markdown("""
                1. **Text Analysis**: The model processes textual content to detect harmful language and context.
                2. **Audio Processing**: Melspectrograms capture audio cues, which are analyzed for tone and 
                   inflection associated with hate speech.
                3. **Fusion Technique**: By combining audio and text modalities, our system provides a more 
                   comprehensive analysis, allowing for nuanced detection across platforms.
            """)
        with col1:
            st.image('assets/image_3.jpeg', width=800, use_container_width=True)
    
    # About Text Modality page
    elif selected == "About Text Modality":
        st.markdown("""<h1 style='text-align: center;'>About Text Modality</h1>""", unsafe_allow_html=True)
        st.markdown("""
            **Text Modality** refers to the analysis of textual content to detect hate speech. 
            We process written data to identify harmful language, context, and intent. This involves 
            techniques such as natural language processing (NLP) and sentiment analysis to detect negativity 
            and harmful content in a piece of text.
        """)
        # Example Graph 1: Distribution of Tweet Lengths
        df = pd.read_csv("assets/data.csv")
        df['tweet_length'] = df['cleaned_tweet'].apply(len)
        st.subheader('Tweet Length Distribution')

        col1, col2 = st.columns([1, 1])
        with col1:
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='tweet_length', hue='is_hate_speech', bins=30, kde=True)
            plt.title('Distribution of Tweet Lengths')
            st.pyplot(plt)
        
        with col2:
            st.subheader("Inference of:")
            st.markdown("""
                This histogram shows the distribution of tweet lengths, differentiated by whether the tweet contains hate speech (is_hate_speech 1) or not (is_hate_speech 0).
Tweets without hate speech (blue bars) tend to be slightly longer on average compared to tweets with hate speech (orange bars).
Both types of tweets peak around the 30-60 character range, but hate speech tweets have a broader range in length.
This suggests that hate speech tweets might be more variable in length, with a higher count at shorter lengths and a lower presence in longer lengths compared to non-hate speech tweets.
            """)
        col1, col2 = st.columns([1, 1])
        # Example Graph 2: Word Cloud for Hate Speech
        hate_speech_text = ' '.join(df[df['is_hate_speech'] == 1]['cleaned_tweet'])

        with col1:
            plt.figure(figsize=(10, 5))
            wordcloud_hate = WordCloud(stopwords=stopwords.words('english'), background_color='black').generate(hate_speech_text)
            plt.imshow(wordcloud_hate, interpolation='bilinear')
            plt.axis("off")
            plt.title('Word Cloud for Hate Speech')
            st.pyplot(plt)

        with col2:
            st.subheader("Inference:")
            st.markdown("""
This word cloud highlights the most frequent words used in tweets labeled as hate speech.
Offensive and derogatory terms dominate, indicating the types of language patterns prevalent in hate speech.
The large size of words like "bitch," "nigga," "faggot," "hoe," "fuck," and "pussy" suggests these terms are frequently used in hate speech tweets.
This visualization provides insight into common language used in hate speech, which could be useful for training machine learning models to identify such content based on vocabulary patterns.





            """)
        col1, col2 = st.columns([1, 1])
        # Example Graph 3: Top Unigrams
        def plot_top_ngrams(corpus, n=1, top_n=20):
            vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
            df_ngram = pd.DataFrame(words_freq, columns=['Ngram', 'Frequency'])
            
            plt.figure(figsize=(10, 5))
            sns.barplot(x='Frequency', y='Ngram', data=df_ngram)
            plt.title(f'Top {n}-grams')
            st.pyplot(plt)
        
        with col1:
            plot_top_ngrams(df[df['is_hate_speech'] == 1]['cleaned_tweet'], n=1)
        
        with col2:
            st.subheader("Inference:")
            st.markdown("""
                The top unigrams for hate speech show frequent use of offensive terms. 
                These words are often short and impactful.
            """)
    # About Audio Modality page
    elif selected == "About Audio Modality":
        st.markdown("""<h1 style='text-align: center;'>About Audio Modality</h1>""", unsafe_allow_html=True)
        st.markdown("""
            **Audio Modality** focuses on analyzing the tonal and acoustic features of speech. 
            We utilize **melspectrograms** to capture the frequency and intensity patterns in spoken 
            words. These patterns are essential for detecting emotions, tones, and cues associated with hate speech.
        """)
        normal_audio_file = "assets/a.wav"
        scream_audio_file = "assets/b.wav"
        compare_audio_files(normal_audio_file, scream_audio_file)

    # Fusion Techniques & Results page
    elif selected == "Fusion Techniques & Results":
        st.markdown("""<h1 style='text-align: center;'>Fusion Techniques & Results</h1>""", unsafe_allow_html=True)
        st.markdown("""
            **Fusion Techniques** combine both audio and text data to enhance the detection of hate speech. 
            Here’s a breakdown of the techniques we use:
            
            - **Early Fusion**: Both text and audio features are combined at the input level and processed 
              together in the same model.
            - **Late Fusion**: Separate models are used for audio and text, and the results are combined 
              at the decision level.
            - **Intermediate Fusion**: A hybrid approach where the intermediate features from both modalities 
              are merged and analyzed together.
            
            **Results**: Our multimodal approach has shown improved accuracy in detecting hate speech 
            compared to traditional unimodal methods. The combination of audio and text data provides a 
            more comprehensive analysis, especially when considering tone and context.
        """)
        data = {
            "Fusion Type": ["Early Fusion", "Hybrid Fusion", "Late Fusion"],
            "Accuracy": [0.546, 0.499, 0.607],
            "Precision": [0.539, 0.499, 0.6],
            "Recall": [0.624, 1.0, 0.635],
            "F1 Score": [0.578, 0.665, 0.617]
        }
        
        # Load data into DataFrame for easy plotting
        df = pd.DataFrame(data)
        
        # Display heading
        st.title("Fusion Model Comparison")
        st.write("Comparison of different fusion model performances across accuracy, precision, recall, and F1 score.")
        
        # Plotting the results
        fig, ax = plt.subplots()
        df.set_index("Fusion Type")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(kind='bar', ax=ax)
        plt.title("Model Comparison by Metrics")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.legend(loc="upper right")
        
        # Display plot in Streamlit
        st.pyplot(fig)
        
        # Inference
        best_model = df.loc[df['F1 Score'].idxmax()]
        st.write("### Inference")
        st.write(f"The best performing fusion type based on the F1 Score is **{best_model['Fusion Type']}**.")
        st.write(f"With an F1 score of **{best_model['F1 Score']}**, this model demonstrates the best balance between precision and recall under the same training parameters for 10 epochs.")
        

if __name__ == "__main__":
    main()



# Multimodal Hate Speech Detection

## Overview

The **Multimodal Hate Speech Detector** is a machine learning application that uses both text and audio data to detect hate speech in online communication. By leveraging **text processing** and **audio processing** (specifically **melspectrograms**), the system is able to classify speech as hate speech or non-hate speech based on the content and tone of the communication.

This project provides a comprehensive analysis of text and audio modalities to offer an improved and nuanced approach to detecting harmful content online. The system is implemented using **Streamlit** for the frontend and uses **librosa** for audio processing and **NLP** techniques for text analysis.

## Features

- **Text Modality**: Analyzes text to detect hate speech through Natural Language Processing (NLP) and sentiment analysis techniques.
- **Audio Modality**: Processes audio data by generating melspectrograms to capture tonal features associated with hate speech.
- **Fusion**: Combines the results from both modalities to provide more accurate hate speech detection.

## Pages

1. **Home**
   - Introduction to hate speech and the system.
   - Explains the multimodal approach using both audio and text data.
   
2. **About Text Modality**
   - Describes the text analysis techniques used to detect harmful language.
   - Visualizations include tweet length distribution and word cloud for hate speech content.
   
3. **About Audio Modality**
   - Explains the role of audio analysis in detecting tone and inflection associated with hate speech.
   
4. **Fusion Techniques & Results**
   - Details the fusion techniques used to combine the text and audio modalities for better accuracy.

## How It Works

### 1. **Text Analysis**
   - The system processes text data (e.g., tweets, comments) to detect harmful language and context.
   - NLP techniques and sentiment analysis are applied to identify negativity or harmful content in the text.

### 2. **Audio Processing**
   - Audio data is processed using **librosa** to generate **melspectrograms**, which capture tonal features like pitch, volume, and cadence—important for detecting spoken hate speech.
   
### 3. **Fusion Approach**
   - The results from both text and audio analysis are combined using **fusion techniques** (Early Fusion, Late Fusion, and Hybrid Fusion).
   - This approach provides a more holistic understanding of the content, improving detection accuracy across platforms.

## Installation

To run the project, you need to have Python installed. You can install the required dependencies using `pip`.

```bash
pip install -r requirements.txt
```

### Dependencies

- Streamlit
- librosa
- matplotlib
- seaborn
- nltk
- pandas
- sklearn
- wordcloud
- streamlit-option-menu

## Usage

Run the Streamlit app with the following command:

```bash
streamlit run app.py
```

This will launch the web application in your browser.

## Visualizations

The following visualizations are generated to provide insights into the dataset and the results of the hate speech detection:

1. **Tweet Length Distribution**: A histogram showing the distribution of tweet lengths, categorized by whether they contain hate speech.
2. **Word Cloud**: A word cloud for hate speech tweets that visualizes the most frequent offensive terms.
3. **Melspectrograms**: Mel spectrogram plots generated for audio data to analyze the tonal features of speech.

## Contributing

Feel free to fork the repository and make improvements! If you find bugs or have suggestions, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **librosa**: For audio processing.
- **Streamlit**: For creating the interactive web application.
- **nltk**: For natural language processing tasks.
- **matplotlib** and **seaborn**: For visualizations.
```

You can modify the details according to your project setup and add any additional sections you might need.

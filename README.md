# StatisticalPlotAnalysis for COSC 524 Natural Language Processing - Project 1


## Project Overview

This project, **Statistical Analysis of Agatha Christie's Novels**, is the first assignment for COSC 524 Natural Language Processing (NLP). The main goal of this project is to analyze the structure, plot progression, and character interactions in crime novels, specifically focusing on those written by Agatha Christie. We utilize various statistical methods and machine learning models to explore and predict key plot elements, such as the primary antagonist (perpetrator) and major turning points within the story.

### Project Components

The project consists of the following main tasks:

1. **Data Exploration**: Loading and preprocessing the novels, splitting the text into chapters, and extracting relevant features from the text (characters, sentiments, crime-related keywords, etc.).
   
2. **Feature Engineering**: Extracting a variety of features related to plot structure, such as character introduction times, co-occurrences between characters, sentiment analysis, and event-based markers.

3. **Statistical Modeling**:
   - **Perpetrator Prediction Model**: Uses features such as character mentions, interactions with the protagonist, and sentiment analysis to predict the primary antagonist of the novel.
   - **Plot Progression Model**: Segments the novel into parts and clusters them into distinct plot events based on features like sentiment shifts and crime-related keywords.

4. **Evaluation**: Cross-validation is performed on the models to ensure generalizability, and the results are evaluated using metrics such as accuracy, precision, recall, and F1-score. The models’ predictions are compared to the actual outcomes.

5. **Visualization**: Character interaction networks, sentiment trends, and plot progression events are visualized to highlight key insights from the novels and the models' outputs.

---

## Requirements

### Software and Libraries

The project is implemented in **Python 3.10** and makes use of the following libraries:
- `spaCy`: For natural language processing, entity recognition, and tokenization.
- `nltk`: For sentiment analysis using the VADER sentiment analyzer and stopword filtering.
- `scikit-learn`: For building machine learning models (Random Forest, KMeans) and performing evaluation through cross-validation.
- `matplotlib` and `seaborn`: For visualizing the results of the analysis, including character interactions and sentiment trends.
- `networkx`: For visualizing character co-occurrence networks.

Install the required libraries using:
```bash
pip install -r requirements.txt

Ensure that the spaCy English model is downloaded:
```bash
python -m spacy download en_core_web_sm

### Files

- **data/:**

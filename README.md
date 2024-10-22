# Statistical Plot Analysis for COSC 524 Natural Language Processing - Project 1


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
```

Ensure that the spaCy English model is downloaded:
```bash
python -m spacy download en_core_web_sm
```

### Files

- **data/:**  Contains the text files of cleaned novels to be analyzed.
- **plots/:** Stores generated plots, such as character interaction networks and sentiment over time.
- **reports/:** Contains detailed analysis reports for each novel, including the results of the perpetrator prediction and plot progression models.
- **analysis/:** Stores a summary of the overall analysis, including predictions and model performance.

### Deliverables

- Python code that performs the analysis and predictions.
- Visualizations of character interactions, sentiment trends, and plot progression.
- Reports for each novel detailing the protagonist, predicted antagonist, major scenes, and plot progression.

---

## How to Run the Project

1. **Prepare the Data:** Place the cleaned text files of the Agatha Christie novels into the `data/` directory. Each file should be a `.txt` file with basic formatting.

2. **Run the Script:**

   - To execute the main script, run the following command in your terminal:
     ```bash
     python main.py
     ```
   - The script will process the novels, extract relevant features, train the models, and generate predictions and visualizations.
  
3. **Generated Outputs:**

   - **Plots:** The generated plots (e.g., character interaction networks, sentiment over time, crime keyword distributions) are saved in the `plots/` directory.
   - **Reports:** Detailed reports for each novel, including the predicted protagonist and antagonist, major scenes, and plot progression, are saved in the `reports/` directory.

---

## Detailed Project Workflow

### Data Exploration

The text of each novel is preprocessed to clean and structure it for analysis. The novel is split into chapters using regular expressions. We then extract various features from the text:

   - **Character Mentions:** Extracts the names of key characters and tracks their first mention and frequency of appearances.
   - **Sentiment Analysis:** Each sentence is analyzed for sentiment using the VADER sentiment analyzer to determine shifts in tone throughout the novel.
   - **Crime-Related Keywords:** The frequency and distribution of crime-related keywords (e.g., "murder", "confession") are tracked to identify key events in the plot.

### Feature Engineering

For each novel, several features are extracted to be used in our models:

   - **Character Features:** Time of first mention, interactions between characters, and co-occurrences in the same scenes.
   - **Sentiment Features:** Sentiment scores associated with characters and scenes.
   - **Event-Based Features:** Crime-related keywords and the locations of their occurrences in the plot.

### Statistical Modeling

1. **Perpetrator Prediction Model:** A supervised machine learning model (Random Forest) is trained to predict the antagonist based on features such as:

   - When and where each character is first mentioned.
   - The number of interactions they have with the protagonist.
   - Sentiment analysis of sentences involving each character.

2. **Plot Progression Model:** This model uses features like sentiment changes, crime keyword frequency, and character mentions to segment the plot into different events and identify key turning points. KMeans clustering is used to group similar events.

### Evaluation and Reporting

The models are evaluated using cross-validation, and metrics such as accuracy, precision, recall, and F1-score are calculated. The results are saved in the `reports/` directory. Additionally, visualizations are created to show:

- **Character Interaction Networks:** Graphs showing which characters interact with each other.
- **Sentiment Trends:** Line plots of sentiment scores throughout the novel.
- **Crime Keyword Distributions:** Histograms showing the distribution of crime-related keywords.

---

## Results

Each report generated contains the following:

- **Protagonist and Antagonist Predictions:** The model's prediction of the antagonist, compared with the actual antagonist from the novel.
- **Major Scenes:** Key scenes identified by the plot progression model based on significant shifts in sentiment and event-based markers.
- **Plot Progression Visualization:** Shows how sentiment and crime-related events change over the course of the novel.

---

## Contact

For any questions or clarifications regarding this project, feel free to reach out to one of the collaborators.

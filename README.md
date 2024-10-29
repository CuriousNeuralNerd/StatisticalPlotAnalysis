# Statistical Plot Analysis for COSC 524 Natural Language Processing - Project 1


## Project Overview

This project, **Statistical Analysis of Agatha Christie's Novels**, is the first assignment for COSC 524 Natural Language Processing (NLP). The main goal of this project is to analyze the structure, plot progression, and character interactions in crime novels, specifically focusing on those written by Agatha Christie. We utilize various statistical methods and machine learning models to explore and predict key plot elements, such as the primary antagonist (perpetrator) and major turning points within the story.

### Project Components

The project consists of the following main tasks:

1. **Data Exploration**: 
   - **Loading and Preprocessing:** Novels are loaded from text files, split into chapters, and preprocessed. We extract relevant features, including character names, sentiments, and crime-related keywords.
   
2. **Feature Engineering**:
   - **Character Features:** Includes the time of first mention, co-occurrences, and interactions between characters.
   - **Sentiment Features:** Sentiment analysis tracks changes in tone associated with characters and scenes.
   - **Event-Based Features:** Includes occurrences of crime-related keywords (e.g., “murder,” “confession”) and reveal-related keywords to identify key moments in the plot.

3. **Statistical Modeling**:
   - **Perpetrator Prediction Model**: A supervised machine learning model (Random Forest) predicts the antagonist based on features such as character mentions, interactions with the protagonist, and sentiment analysis.
   - **Plot Progression Model**: This model segments the novel into events and identifies plot progression by clustering similar sections based on sentiment shifts, crime-related keywords, and reveal keywords. KMeans clustering is used to group events, while a neural network (MLP) evaluates the reveal point.

4. **Evaluation**:
   - Antagonist Prediction Model: Evaluated using cross-validation with metrics such as accuracy, precision, recall, and F1-score.
   - Plot Progression Model: Evaluated using cross-validation with Mean Squared Error (MSE) to assess accuracy in predicting reveal points.

5. **Visualization**: 
   - Character interaction networks, sentiment trends, and plot progression events are visualized to highlight insights from the novels and the models' outputs.

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
- **plots/:** Stores generated plots, including character interaction networks, sentiment over time, and keyword distributions.
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

   - **Plots:** The generated plots (e.g., sentiment over time, keyword frequency distributions) are saved in the `plots/` directory.
   - **Reports:** Detailed reports for each novel, including the predicted protagonist and antagonist, major scenes, and plot progression, are saved in the `reports/` directory.

---

## Detailed Project Workflow

### Data Exploration

Each novel's text undergoes preprocessing to clean and structure it for analysis. The text is split into chapters and then into sentences, with the following key elements extracted:

   - **Character Mentions:** Key characters are identified and standardized, with their first mention and frequency tracked across the text.
   - **Sentiment Analysis:** Each sentence is analyzed for sentiment using the VADER sentiment analyzer to determine shifts in tone throughout the novel.
   - **Crime and Reveal Keywords:** The frequency and distribution of the keywords (e.g., "murder", "reveal") are tracked to identify key events in the plot.

### Feature Engineering

For each novel, several features are extracted to be used in our models:

   - **Character Features:** Time of first mention, interactions between characters, and co-occurrences in the same scenes.
   - **Sentiment Features:** Sentiment scores are calculated per sentence and associated with characters and scenes.
   - **Event-Based Features:** Frequency and locations of crime-related and reveal-related keywords are tracked across chapters.

### Statistical Modeling

1. **Perpetrator Prediction Model:** A supervised machine learning model (Random Forest) is trained to predict the antagonist based on features such as:

   - **Character Sentiment:** Tracks the sentiment associated with each character, as antagonists may have distinctive sentiment patterns.
   - **Crime Keyword Co-occurrence:** Measures how often a character appears in sentences with crime-related keywords, indicating involvement in key plot moments.
   - **First Mention Position:** Records the chapter and sentence index of each character’s first mention, as early introductions can signal importance in the plot.
   - **Protagonist Interactions:** Counts interactions with the protagonist, as antagonists often have significant or adversarial connections with them.
   - **Network Centrality:** Calculates each character's centrality within the interaction network, where higher centrality might imply greater narrative importance.
     
The model is evaluated using cross-validation metrics like accuracy, precision, recall, and F1-score.
   
2. **Plot Progression Model:** This model uses features like sentiment changes, crime and reveal keyword frequency, and character mentions to segment the plot into different events and identify key turning points.
   
   - **Clustering:** Uses KMeans to identify plot segments based on sentiment, character mentions, and keyword frequencies.
   - **Neural Network (MLP):** Predicts the key reveal point by assessing segment features against known reveal moments

### Evaluation and Reporting

Models are evaluated as follows:

   - **Antagonist Prediction Model:** Cross-validated using metrics such as accuracy, precision, recall, and F1-score. Results are saved in the reports/ directory.
   - **Plot Progression Model:** Evaluated with cross-validation using Mean Squared Error (MSE).

Visualizations are created for:

   - **Character Interaction Networks:** Graphs displaying character co-occurrences.
   - **Sentiment Trends:** Line plots of sentiment scores throughout each novel.
   - **Crime and Reveal Keyword Distributions:** Histograms showing the distribution of these keywords across the text.

---

## Results

Each report generated contains the following:

- **Protagonist and Antagonist Predictions:** The model's prediction of the antagonist, compared with the actual antagonist from the novel.
- **Major Scenes:** Key scenes identified by the plot progression model based on significant shifts in sentiment and event-based markers.
- **Plot Progression Visualization:** Shows how sentiment and crime and reveal-related events change over the course of the novel.

---

## Contact

For any questions or clarifications regarding this project, feel free to reach out to one of the collaborators.

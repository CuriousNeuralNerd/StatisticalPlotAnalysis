# Import necessary libraries
import os
import re
import spacy
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Enable GPU support for spaCy
spacy.require_gpu()

# Download necessary NLTK data files
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
sia = SentimentIntensityAnalyzer()

# Set random state for reproducibility
rState = 42

# Create directories for outputs
os.makedirs('plots', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('analysis', exist_ok=True)

# Load spaCy English model and disable unnecessary components
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Add the sentencizer to the pipeline
nlp.add_pipe('sentencizer')

# Data Exploration and Preprocessing
def load_and_preprocess_text(file_path):
    """
    Reads a cleaned text file and processes the text using spaCy.
    Splits the text into chapters.

    Args:
        file_path (str): Path to the cleaned text file.

    Returns:
        chapters (list): List of dictionaries containing chapter titles and text.
    """
    # Read the cleaned text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Basic cleaning
    text = text.strip()

    # Split text into chapters using regular expressions
    chapter_pattern = re.compile(r'(Chapter|CHAPTER)\s+([^\n]+)\n', re.IGNORECASE)
    chapters = []
    matches = list(chapter_pattern.finditer(text))
    chapter_starts = [match.start() for match in matches]
    chapter_titles = [match.group().strip() for match in matches]

    if chapter_starts:
        for idx in range(len(chapter_starts)):
            start = chapter_starts[idx]
            end = chapter_starts[idx + 1] if idx + 1 < len(chapter_starts) else len(text)
            chapter_text = text[start:end].strip()
            chapter_title = chapter_titles[idx]
            chapters.append({'title': chapter_title, 'text': chapter_text})
    else:
        # If no chapters are found, treat the entire text as one chapter
        chapters.append({'title': 'Chapter 1', 'text': text})

    return chapters

# List of novels to analyze
novel_paths = [
    'data/clean_novel1.txt', 'data/clean_novel2.txt',
    'data/clean_novel4.txt', 'data/clean_novel5.txt', 'data/clean_novel6.txt',
    'data/clean_novel7.txt', 'data/clean_novel8.txt', 'data/clean_novel9.txt',
]

# Placeholder for novel data
novels_data = []

# Process each novel
for path in novel_paths:
    chapters = load_and_preprocess_text(path)
    novels_data.append({
        'title': os.path.basename(path).replace('.txt', ''),
        'chapters': chapters
    })

# List of crime-related keywords
crime_keywords = [
    'murder', 'investigation', 'guilt', 'confession', 'theft', 'punishment',
    'revenge', 'deception', 'violence', 'betrayal', 'blackmail', 'suspect',
    'crime', 'police', 'escape', 'lie', 'disguise', 'alibi', 'false accusation'
]

# Stop words for filtering
stop_words = set(stopwords.words('english'))

def analyze_novel(novel):
    """
    Analyzes a novel to extract various features such as major characters, sentiments,
    interactions, plot structure, and more.

    Args:
        novel (dict): A dictionary containing the novel's title and chapters.

    Returns:
        analysis (dict): A dictionary containing analysis results for the novel.
    """
    analysis = {}  # Initialize the analysis dictionary
    title = novel['title']
    chapters = novel['chapters']

    # Character aliases mapping to standardize names
    character_aliases = {
        # The Murder of Roger Ackroyd
        'Hercule Poirot': ['Hercule Poirot', 'Poirot', 'Hercule', 'Mr. Poirot'],
        'Dr. James Sheppard': ['Dr. James Sheppard', 'Sheppard', 'James', 'Doctor', 'Dr. Sheppard', 'The doctor', 'The narrator'],
        'Roger Ackroyd': ['Ackroyd', 'Roger', 'Mr. Ackroyd'],
        'Ralph Paton': ['Ralph', 'Captain Paton', 'Mr. Paton'],
        'Flora Ackroyd': ['Flora', 'Miss Ackroyd'],
        'Ursula Bourne': ['Ursula', 'Miss Bourne', 'Mrs. Paton'],
        'Parker': ['Parker'],
        'Inspector Raglan': ['Raglan', 'The inspector'],
        # The Mysterious Affair at Styles
        'Emily Inglethorp': ['Emily Inglethorp', 'Mrs. Inglethorp', 'Emily'],
        'Alfred Inglethorp': ['Alfred Inglethorp', 'Alfred', 'Mr. Inglethorp'],
        # The Murder on the Links
        'Jack Renauld': ['Jack', 'Mr. Renauld'],
        'Paul Renauld': ['Paul Renauld', 'Paul', 'Mr. Renauld', 'Georges Conneau'],
        'Madame Daubreuil': ['Madame Daubreuil', 'Madame Beroldy', 'Marthe’s mother'],
        'Marthe Daubreuil': ['Marthe Daubreuil', 'Marthe', 'Miss Daubreuil'],
        # The Man in the Brown Suit
        'Anne Beddingfeld': ['Anne Beddingfeld', 'Anne', 'Miss Beddingfeld'],
        'Sir Eustace Pedler': ['Sir Eustace Pedler', 'Sir Eustace', 'Pedler', 'The Colonel'],
        # The Mystery of the Blue Train
        'Major Richard Knighton': ['Major Richard Knighton', 'Knighton', 'Major Knighton'],
        # The Big Four
        'Captain Hastings': ['Hastings', 'Captain Hastings'],
        'Claude Darrell': ['The Destroyer', 'Number Four', 'Claude Darrell'],
        # The Secret Adversary
        'Tuppence Cowley': ['Tuppence Cowley', 'Tuppence', 'Miss Cowley', 'Prudence Cowley'],
        'Mr. Brown': ['Mr. Brown', 'Sir James Peel Edgerton', 'The secret adversary'],
        # The Secret of Chimneys
        'Anthony Cade': ['Anthony Cade', 'Anthony', 'Mr. Cade', 'James McGrath'],
        'King Victor': ['Baron Lolopretjzyl', 'King Victor', 'Victor', 'The Baron'],
    }

    # Map character aliases to standard names
    alias_to_standard = {}
    for standard_name, aliases in character_aliases.items():
        for alias in aliases:
            alias_to_standard[alias.lower()] = standard_name

    def standardize_name(name):
        return alias_to_standard.get(name.lower(), name.strip())

    # Initialize overall data structures
    overall_character_entities = []
    overall_sentiments = []
    overall_sentences = []
    major_scenes = []
    character_first_mention = {}
    crime_first_mention = None
    crime_keyword_positions = []

    # Character interaction co-occurrence matrix
    co_occurrence_counts = defaultdict(int)

    # Process all chapters at once using nlp.pipe
    chapter_texts = [chapter['text'] for chapter in chapters]
    docs = list(nlp.pipe(chapter_texts))

    cumulative_sentences = 0
    cumulative_sentence_counts = []

    for chapter_idx, (chapter, doc) in enumerate(zip(chapters, docs)):
        chapter_title = chapter['title']

        # Extract sentences
        sentences = list(doc.sents)
        overall_sentences.extend(sentences)
        cumulative_sentence_counts.append(len(sentences))

        # Extract named entities and map to standardized character names
        character_entities = []
        sent_characters = [set() for _ in sentences]
        for sent_idx, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            # Extract characters in the sentence
            # Use regex to find capitalized words as potential names
            person_names = re.findall(r'\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)*', sent_text)
            sent_ents = [standardize_name(name) for name in person_names]
            sent_chars = set(name for name in sent_ents if name.lower() not in stop_words)

            # Record first mention position for each character
            for char in sent_chars:
                if char not in character_first_mention:
                    character_first_mention[char] = {
                        'chapter': chapter_idx + 1,
                        'sentence': cumulative_sentences + sent_idx + 1
                    }

            character_entities.extend(sent_chars)
            sent_characters[sent_idx] = sent_chars

            # Check for crime-related keywords
            if any(crime_word in sent_text.lower() for crime_word in crime_keywords):
                # Record first mention of crime
                if crime_first_mention is None:
                    crime_first_mention = {
                        'chapter': chapter_idx + 1,
                        'sentence': cumulative_sentences + sent_idx + 1
                    }
                crime_keyword_positions.append(cumulative_sentences + sent_idx + 1)

            # Compute sentiment for the sentence
            sentiment = sia.polarity_scores(sent_text)['compound']
            overall_sentiments.append(sentiment)

        # Character co-occurrence
        for sent_chars in sent_characters:
            chars = list(sent_chars)
            for i in range(len(chars)):
                for j in range(i + 1, len(chars)):
                    pair = tuple(sorted([chars[i], chars[j]]))
                    co_occurrence_counts[pair] += 1

        overall_character_entities.extend(character_entities)
        cumulative_sentences += len(sentences)

    # Count character occurrences
    overall_character_counts = Counter(overall_character_entities)
    # Set a frequency threshold to filter out incidental mentions and misclassifications
    frequency_threshold = 5
    overall_major_characters = [char for char, count in overall_character_counts.items() if count >= frequency_threshold]

    # If no major characters found, use top N characters
    if not overall_major_characters:
        overall_major_characters = [char for char, count in overall_character_counts.most_common(5)]

    # Analyze sentiments associated with each major character
    character_sentiments = {}
    char_to_sent_indices = defaultdict(list)
    for idx, sent in enumerate(overall_sentences):
        sent_text = sent.text.strip()
        # Extract characters in the sentence
        person_names = re.findall(r'\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)*', sent_text)
        sent_ents = [standardize_name(name) for name in person_names]
        sent_chars = [name for name in sent_ents if name in overall_major_characters]
        for char in sent_chars:
            char_to_sent_indices[char].append(idx)

    for char in overall_major_characters:
        indices = char_to_sent_indices[char]
        char_sentiments = [overall_sentiments[i] for i in indices]
        character_sentiments[char] = np.mean(char_sentiments) if char_sentiments else 0

    # Compute co-occurrence with crime-related keywords
    character_crime_cooccurrence = {}
    crime_sent_indices = set(crime_keyword_positions)
    for char in overall_major_characters:
        char_sent_indices = set(char_to_sent_indices[char])
        cooccurrence_count = len(char_sent_indices & crime_sent_indices)
        character_crime_cooccurrence[char] = cooccurrence_count

    # Get the novel's filename without extension
    novel_filename = title

    # Predefined mappings for protagonists and antagonists
    novel_roles = {
        'clean_novel1': {
            'protagonist': 'Hercule Poirot',
            'antagonist': 'Dr. James Sheppard'
        },
        'clean_novel2': {
            'protagonist': 'Hercule Poirot',
            'antagonist': 'Alfred Inglethorp'
        },
        'clean_novel4': {
            'protagonist': 'Hercule Poirot',
            'antagonist': 'Marthe Daubreuil'
        },
        'clean_novel5': {
            'protagonist': 'Anne Beddingfeld',
            'antagonist': 'Sir Eustace Pedler'
        },
        'clean_novel6': {
            'protagonist': 'Hercule Poirot',
            'antagonist': 'Major Richard Knighton'
        },
        'clean_novel7': {
            'protagonist': 'Hercule Poirot',
            'antagonist': 'Claude Darrell'
        },
        'clean_novel8': {
            'protagonist': 'Tuppence Cowley',
            'antagonist': 'Mr. Brown'
        },
        'clean_novel9': {
            'protagonist': 'Anthony Cade',
            'antagonist': 'King Victor'
        }
    }

    # Use predefined protagonist and antagonist
    protagonist = novel_roles.get(novel_filename, {}).get('protagonist')
    antagonist = novel_roles.get(novel_filename, {}).get('antagonist')

    # Additional features for modeling
    # Position of first appearance of antagonist and protagonist
    antagonist_first_mention = character_first_mention.get(antagonist, {'chapter': np.nan, 'sentence': np.nan})
    protagonist_first_mention = character_first_mention.get(protagonist, {'chapter': np.nan, 'sentence': np.nan})

    # Number of interactions with antagonist/protagonist
    interaction_with_protagonist = co_occurrence_counts.get(tuple(sorted([protagonist, antagonist])), 0)
    interaction_with_others = sum([co_occurrence_counts.get(tuple(sorted([protagonist, other])), 0)
                                   for other in overall_major_characters if other != protagonist])

    # Network centrality measures
    # Build character interaction graph
    G = nx.Graph()
    for pair, weight in co_occurrence_counts.items():
        G.add_edge(pair[0], pair[1], weight=weight)
    if G.has_node(antagonist):
        antagonist_centrality = nx.degree_centrality(G)[antagonist]
    else:
        antagonist_centrality = 0
    if G.has_node(protagonist):
        protagonist_centrality = nx.degree_centrality(G)[protagonist]
    else:
        protagonist_centrality = 0

    # Identify major scenes based on significant sentiment changes
    sentiment_magnitude = [abs(score) for score in overall_sentiments]
    mean_magnitude = np.mean(sentiment_magnitude)
    std_magnitude = np.std(sentiment_magnitude)
    threshold = mean_magnitude + std_magnitude

    major_scene_indices = [i for i, mag in enumerate(sentiment_magnitude) if mag >= threshold]

    # Record major scenes with chapter and percentage through the novel
    cumulative_sentence_counts = np.cumsum([0] + cumulative_sentence_counts)
    for scene_idx in major_scene_indices:
        chapter_idx = np.searchsorted(cumulative_sentence_counts, scene_idx, side='right') - 1
        sentence_position = ((scene_idx - cumulative_sentence_counts[chapter_idx]) /
                             (cumulative_sentence_counts[chapter_idx + 1] - cumulative_sentence_counts[chapter_idx])) * 100
        major_scenes.append({
            'chapter': chapters[chapter_idx]['title'],
            'chapter_number': chapter_idx + 1,
            'scene_sentence': overall_sentences[scene_idx].text.strip(),
            'position_in_chapter': sentence_position
        })

    # Plot Progression Model Implementation
    # -------------------------------------
    # Segment the novel into equal parts
    num_segments = 10  # You can adjust the number of segments
    total_sentences = len(overall_sentences)
    segment_size = total_sentences // num_segments

    segments = []
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else total_sentences
        segment_sentences = overall_sentences[start_idx:end_idx]
        segment_text = ' '.join([sent.text for sent in segment_sentences])

        # Extract features for the segment
        # Average sentiment
        segment_sentiments = overall_sentiments[start_idx:end_idx]
        avg_sentiment = np.mean(segment_sentiments) if segment_sentiments else 0

        # Frequency of crime-related keywords
        segment_text_lower = segment_text.lower()
        crime_keyword_count = sum(segment_text_lower.count(kw) for kw in crime_keywords)

        # Number of character mentions
        # Use regex for person names
        person_names = re.findall(r'\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)*', segment_text)
        segment_characters = [standardize_name(name) for name in person_names]
        segment_character_count = len(segment_characters)

        # Store the features
        segments.append({
            'segment_index': i,
            'start_sentence': start_idx,
            'end_sentence': end_idx,
            'avg_sentiment': avg_sentiment,
            'crime_keyword_count': crime_keyword_count,
            'character_mention_count': segment_character_count
        })

    # Use KMeans clustering to cluster segments into plot events
    # (This is a simplification due to lack of labeled data)
    segment_features = pd.DataFrame(segments)
    X_segments = segment_features[['avg_sentiment', 'crime_keyword_count', 'character_mention_count']]
    num_clusters = 5  # Assuming 5 different plot events
    kmeans = KMeans(n_clusters=num_clusters, random_state=rState, n_init=10)
    segment_features['plot_event'] = kmeans.fit_predict(X_segments)

    # Add plot progression data to analysis
    analysis['plot_progression'] = segment_features

    # Compile analysis results
    analysis.update({
        'title': novel_filename,
        'major_characters': overall_major_characters,
        'protagonist': protagonist,
        'antagonist': antagonist,
        'character_sentiments': character_sentiments,
        'character_crime_cooccurrence': character_crime_cooccurrence,
        'major_scenes': major_scenes,
        'character_first_mention': character_first_mention,
        'crime_first_mention': crime_first_mention,
        'co_occurrence_counts': co_occurrence_counts,
        'antagonist_first_mention': antagonist_first_mention,
        'protagonist_first_mention': protagonist_first_mention,
        'interaction_with_protagonist': interaction_with_protagonist,
        'interaction_with_others': interaction_with_others,
        'antagonist_centrality': antagonist_centrality,
        'protagonist_centrality': protagonist_centrality,
        'overall_sentiments': overall_sentiments,
        'chapters': chapters,
        'crime_keyword_positions': crime_keyword_positions
    })
    return analysis

# Analyze each novel and store the analyses
analyses = []
for novel in tqdm(novels_data, desc="Analyzing Novels"):
    analysis = analyze_novel(novel)
    analyses.append(analysis)

# Prepare data for antagonist and protagonist prediction
def prepare_data_for_modeling(analyses, role='antagonist'):
    """
    Prepares feature matrix X and label vector y for the antagonist/protagonist prediction model.

    Args:
        analyses (list): List of analysis dictionaries for each novel.
        role (str): 'antagonist' or 'protagonist'.

    Returns:
        X (DataFrame): Feature matrix.
        y (Series): Label vector.
        titles (list): List of novel titles.
    """
    features = []
    labels = []
    titles = []

    for analysis in analyses:
        if analysis['protagonist'] and analysis['antagonist']:
            character = analysis[role]
            features.append([
                len(analysis['major_characters']),
                analysis['character_sentiments'].get(character, 0),
                analysis['character_crime_cooccurrence'].get(character, 0),
                analysis[f'{role}_first_mention']['sentence'],  # Position of first mention
                analysis['interaction_with_protagonist'] if role == 'antagonist' else analysis['interaction_with_others'],
                analysis[f'{role}_centrality']                  # Centrality measure
            ])
            labels.append(character)
            titles.append(analysis['title'])

    X = pd.DataFrame(features, columns=[
        'num_characters',
        f'{role}_sentiment',
        'crime_cooccurrence',
        f'{role}_first_mention',
        f'interaction_with_{"protagonist" if role == "antagonist" else "others"}',
        f'{role}_centrality'
    ])
    y = pd.Series(labels)
    return X, y, titles

# Train Random Forest models to predict the antagonist and protagonist
# Antagonist Model
X_ant, y_ant, titles_ant = prepare_data_for_modeling(analyses, role='antagonist')
label_encoder_ant = LabelEncoder()
y_ant_encoded = label_encoder_ant.fit_transform(y_ant)
rf_model_ant = RandomForestClassifier(n_estimators=100, random_state=rState)

if len(X_ant) >= 2:
    n_splits = min(5, len(X_ant))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rState)
    accuracies = cross_val_score(rf_model_ant, X_ant, y_ant_encoded, cv=kf, scoring='accuracy')
    print(f"Antagonist Model Cross-Validation Accuracy: {np.mean(accuracies):.4f}")

rf_model_ant.fit(X_ant, y_ant_encoded)
y_ant_pred_encoded = rf_model_ant.predict(X_ant)
y_ant_pred = label_encoder_ant.inverse_transform(y_ant_pred_encoded)

# Protagonist Model
X_prot, y_prot, titles_prot = prepare_data_for_modeling(analyses, role='protagonist')
label_encoder_prot = LabelEncoder()
y_prot_encoded = label_encoder_prot.fit_transform(y_prot)
rf_model_prot = RandomForestClassifier(n_estimators=100, random_state=rState)

if len(X_prot) >= 2:
    n_splits = min(5, len(X_prot))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rState)
    accuracies = cross_val_score(rf_model_prot, X_prot, y_prot_encoded, cv=kf, scoring='accuracy')
    print(f"Protagonist Model Cross-Validation Accuracy: {np.mean(accuracies):.4f}")

rf_model_prot.fit(X_prot, y_prot_encoded)
y_prot_pred_encoded = rf_model_prot.predict(X_prot)
y_prot_pred = label_encoder_prot.inverse_transform(y_prot_pred_encoded)

# Output the novel titles with their predicted antagonists and protagonists
for analysis, antagonist_pred, protagonist_pred in zip(analyses, y_ant_pred, y_prot_pred):
    title = analysis['title']
    # Prepare report content
    report_content = f"Novel: {title}\n"
    report_content += f"Predicted Protagonist: {protagonist_pred}\n"
    report_content += f"Predicted Antagonist: {antagonist_pred}\n"
    report_content += "-" * 50 + "\n"

    # Include major scenes
    report_content += "Major Scenes:\n"
    for scene in analysis['major_scenes']:
        report_content += f"  Chapter: {scene['chapter']} (Chapter {scene['chapter_number']})\n"
        report_content += f"  Position in Chapter: {scene['position_in_chapter']:.2f}%\n"
        report_content += f"  Scene Sentence: {scene['scene_sentence']}\n\n"
    report_content += "-" * 50 + "\n"

    # Include plot progression analysis
    plot_progression = analysis['plot_progression']
    report_content += "Plot Progression:\n"
    report_content += plot_progression[['segment_index', 'avg_sentiment', 'crime_keyword_count', 'character_mention_count', 'plot_event']].to_string(index=False)
    report_content += "\n" + "=" * 50 + "\n"

    # Write report to file
    report_filename = os.path.join('reports', f"{title}_report.txt")
    with open(report_filename, 'w') as f:
        f.write(report_content)

# Print Classification Reports
print("Antagonist Classification Report:")
report_ant = classification_report(y_ant, y_ant_pred, target_names=label_encoder_ant.classes_)
print(report_ant)
# Save report to file
report_filename_ant = os.path.join('reports', 'antagonist_classification_report.txt')
with open(report_filename_ant, 'w') as f:
    f.write("Antagonist Classification Report:\n")
    f.write(report_ant)

print("Protagonist Classification Report:")
report_prot = classification_report(y_prot, y_prot_pred, target_names=label_encoder_prot.classes_)
print(report_prot)
# Save report to file
report_filename_prot = os.path.join('reports', 'protagonist_classification_report.txt')
with open(report_filename_prot, 'w') as f:
    f.write("Protagonist Classification Report:\n")
    f.write(report_prot)

# Generate Visualizations
# Character Interaction Network
for analysis in analyses:
    title = analysis['title']
    co_occurrence_counts = analysis['co_occurrence_counts']
    G = nx.Graph()
    for pair, weight in co_occurrence_counts.items():
        if weight > 0:
            G.add_edge(pair[0], pair[1], weight=weight)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=rState)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, width=[v['weight'] for (u, v, v) in G.edges(data=True)])
    nx.draw_networkx_labels(G, pos)
    plt.title(f"Character Interaction Network for {title}")
    # Save plot
    plot_filename = os.path.join('plots', f"{title}_interaction_network.png")
    plt.savefig(plot_filename)
    plt.close()

# Sentiment Over Time Plot
for analysis in analyses:
    title = analysis['title']
    sentiments = analysis['overall_sentiments']
    plt.figure(figsize=(12, 6))
    plt.plot(sentiments)
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentiment Score')
    plt.title(f"Sentiment Over Time for {title}")
    # Save plot
    plot_filename = os.path.join('plots', f"{title}_sentiment_over_time.png")
    plt.savefig(plot_filename)
    plt.close()

# Crime-related Keyword Frequency Distribution
for analysis in analyses:
    title = analysis['title']
    crime_positions = analysis.get('crime_keyword_positions', [])
    plt.figure(figsize=(12, 6))
    sns.histplot(crime_positions, bins=20, kde=False)
    plt.xlabel('Sentence Index')
    plt.ylabel('Frequency')
    plt.title(f"Crime-related Keyword Frequency Distribution for {title}")
    # Save plot
    plot_filename = os.path.join('plots', f"{title}_crime_keyword_distribution.png")
    plt.savefig(plot_filename)
    plt.close()

# Plot Progression Visualization
for analysis in analyses:
    title = analysis['title']
    plot_progression = analysis['plot_progression']
    plt.figure(figsize=(12, 6))
    plt.plot(plot_progression['segment_index'], plot_progression['avg_sentiment'], marker='o', label='Average Sentiment')
    plt.bar(plot_progression['segment_index'], plot_progression['crime_keyword_count'], alpha=0.5, label='Crime Keyword Count')
    plt.xlabel('Segment Index')
    plt.ylabel('Value')
    plt.title(f"Plot Progression Analysis for {title}")
    plt.legend()
    # Save plot
    plot_filename = os.path.join('plots', f"{title}_plot_progression.png")
    plt.savefig(plot_filename)
    plt.close()

# Analysis
# -----------------------

# Save analysis outputs to a file
analysis_filename = os.path.join('analysis', 'analysis.txt')
with open(analysis_filename, 'w') as f:
    f.write("\nAnalysis:\n")
    for i, analysis in enumerate(analyses):
        title = analysis['title']
        actual_antagonist = y_ant.iloc[i]
        predicted_antagonist = y_ant_pred[i]
        actual_protagonist = y_prot.iloc[i]
        predicted_protagonist = y_prot_pred[i]
        f.write(f"Novel: {title}\n")
        f.write(f"Actual Protagonist: {actual_protagonist}\n")
        f.write(f"Predicted Protagonist: {predicted_protagonist}\n")
        if actual_protagonist == predicted_protagonist:
            f.write("The model correctly identified the protagonist.\n")
        else:
            f.write("The model failed to identify the correct protagonist.\n")
        f.write(f"Actual Antagonist: {actual_antagonist}\n")
        f.write(f"Predicted Antagonist: {predicted_antagonist}\n")
        if actual_antagonist == predicted_antagonist:
            f.write("The model correctly identified the antagonist.\n")
        else:
            f.write("The model failed to identify the correct antagonist.\n")
        f.write("-" * 50 + "\n")
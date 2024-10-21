import re

def clean_novel(novel_text):
    """
    Cleans the novel text by:
    - Removing all content before the "*** Start" marker and up to the next "***".
    - Replacing Roman numerals beside "CHAPTER" with the corresponding numbers.
    - Removing all content after the "*** THE END ***" marker.

    Args:
        novel_text (str): The raw text of the novel.

    Returns:
        str: The cleaned text of the novel.
    """
    # Step 1: Remove everything before "*** START ..." and up to the next "***"
    # This will match everything before "*** START" and remove until the next "***".
    start_pattern = r".*?\*\*\* START.*?\*\*\*"
    cleaned_text = re.sub(start_pattern, '', novel_text, count=1, flags=re.IGNORECASE | re.DOTALL)

    # Step 2: Replace Roman numerals with numbers for "CHAPTER" headings
    roman_to_num = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25, 
        'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30,
        'XXXI': 31, 'XXXII': 32, 'XXXIII': 33, 'XXXIV': 34, 'XXXV': 35,
        'XXXVI': 36, 'XXXVII': 37, 'XXXVIII': 38, 'XXXIX': 39, 'XL': 40,
    }

    def replace_roman(match):
        chapter_num = match.group(1)
        return f"CHAPTER {roman_to_num.get(chapter_num, chapter_num)}"
    
     # This regex now matches Roman numerals up to "XL" (40)
    chapter_pattern = r"CHAPTER (X{0,3}(IX|IV|V?I{0,3}))"
    cleaned_text = re.sub(chapter_pattern, replace_roman, cleaned_text)

    # Step 3: Remove everything after "*** THE END ***"
    end_pattern = r"\*\*\* THE END \*\*\*.*"
    cleaned_text = re.sub(end_pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)

    return cleaned_text


def process_and_save_novels(input_paths, output_paths):
    """
    Processes and cleans the novel text from input files and saves the cleaned text to output files.

    Args:
        input_paths (list of str): List of file paths of the raw novel text files.
        output_paths (list of str): List of file paths where the cleaned novels should be saved.
    """
    for input_path, output_path in zip(input_paths, output_paths):
        # Read the novel text from the file
        with open(input_path, 'r', encoding='utf-8') as f:
            novel_text = f.read()

        # Clean the novel text
        cleaned_text = clean_novel(novel_text)

        # Save the cleaned text to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        print(f'Cleaned and saved: {output_path}')


novel_input_paths = [
    'data/novel1.txt', 'data/novel2.txt', 'data/novel3.txt', 
    'data/novel4.txt', 'data/novel5.txt', 'data/novel6.txt',
    'data/novel7.txt', 'data/novel8.txt', 'data/novel9.txt',
    'data/novel10.txt', 'data/novel11.txt', 'data/novel12.txt'
]

novel_output_paths = [
    'data/clean_novel1.txt', 'data/clean_novel2.txt', 'data/clean_novel3.txt',
    'data/clean_novel4.txt', 'data/clean_novel5.txt', 'data/clean_novel6.txt',
    'data/clean_novel7.txt', 'data/clean_novel8.txt', 'data/clean_novel9.txt',
    'data/clean_novel10.txt', 'data/clean_novel11.txt', 'data/clean_novel12.txt'
]

process_and_save_novels(novel_input_paths, novel_output_paths)
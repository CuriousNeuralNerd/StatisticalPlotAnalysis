import os
import requests


def download_novel(url, save_path):
    """
    Downloads the text of a novel from the given URL and saves it to the specified path.
    
    Args:
        url (str): The URL of the novel.
        save_path (str): The local path where the novel will be saved.
    """
    response = requests.get(url)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

# List of URLs and corresponding local paths
novel_urls = [
    'https://www.gutenberg.org/cache/epub/69087/pg69087.txt',
    'https://www.gutenberg.org/cache/epub/863/pg863.txt',
    'https://www.gutenberg.org/cache/epub/61262/pg61262.txt',
    'https://www.gutenberg.org/cache/epub/58866/pg58866.txt',
    'https://www.gutenberg.org/cache/epub/61168/pg61168.txt',
    'https://www.gutenberg.org/cache/epub/72824/pg72824.txt',
    'https://www.gutenberg.org/cache/epub/70114/pg70114.txt',
    'https://www.gutenberg.org/cache/epub/1155/pg1155.txt',
    'https://www.gutenberg.org/cache/epub/65238/pg65238.txt',
    'https://www.gutenberg.org/cache/epub/67173/pg67173.txt',
    'https://www.gutenberg.org/cache/epub/66446/pg66446.txt',
    'https://www.gutenberg.org/cache/epub/67160/pg67160.txt'
]

# Corresponding local paths
novel_save_paths = [
    'data/novel1.txt', 'data/novel2.txt', 'data/novel3.txt', 
    'data/novel4.txt', 'data/novel5.txt', 'data/novel6.txt',
    'data/novel7.txt', 'data/novel8.txt', 'data/novel9.txt',
    'data/novel10.txt', 'data/novel11.txt', 'data/novel12.txt'
]


# Download each novel
for url, save_path in zip(novel_urls, novel_save_paths):
    download_novel(url, save_path)
    print(f'Downloaded and saved: {save_path}')
import gzip
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

# Initialize Stopwords and Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Function to clean text (remove URLs, mentions, hashtags, and special characters)
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z0-9\s?]', '', text)  # Keep question marks, remove other non-alphabetical characters
    text = re.sub(r"['\[\]]", '', text)  # Remove quotes and square brackets
    return text


# Function to tokenize and remove stopwords
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    return [word for word in tokens if word.lower() not in stop_words]


# Function to map POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Function to lemmatize tokens with POS tagging
def lemmatize_tokens(tokens):
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]


# Function to remove illegal characters from the DataFrame text
def remove_illegal_characters(text):
    if isinstance(text, str):  # Check if the value is a string
        # Remove non-ASCII characters (i.e., any character not in the ASCII range)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Remove control characters (ASCII 0x00-0x1F) which are illegal in Excel
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text


def count_records_by_category(df, category_column='issue_label'):
    if category_column in df.columns:
        category_counts = df[category_column].value_counts()
        print("\nRecord counts by category:")
        for category, count in category_counts.items():
            print(f"{category}: {count}")
    else:
        print(f"No '{category_column}' column found for counting categories.")


# Main preprocessing function
def preprocess_text(input_data):
    if isinstance(input_data, str):
        # Handle single string input
        text = input_data.lower()
        text = clean_text(text)
        tokens = tokenize_and_remove_stopwords(text)
        tokens = lemmatize_tokens(tokens)
        return tokens

    elif isinstance(input_data, list) or isinstance(input_data, pd.DataFrame):
        df = pd.DataFrame(input_data)
        if 'issue_title' in df.columns and 'issue_body' in df.columns:
            df['issue_title'] = df['issue_title'].fillna('')
            df['issue_body'] = df['issue_body'].fillna('')
        elif 'title' in df.columns and 'body' in df.columns:
            df['issue_title'] = df['title'].fillna('')
            df['issue_body'] = df['body'].fillna('')
        else:
            df['issue_title'] = ''
            df['issue_body'] = ''

        # Combine `title` and `body` into `text`
        df['text'] = df['issue_title'] + ' ' + df['issue_body']

        # Normalize and clean the text
        df['text'] = df['text'].str.encode('ascii', 'ignore').str.decode('ascii')
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].apply(clean_text)

        # Tokenize, remove stopwords, and lemmatize
        df['tokens'] = df['text'].apply(tokenize_and_remove_stopwords)
        df['tokens'] = df['tokens'].apply(lemmatize_tokens)

        # Remove illegal characters from the entire DataFrame
        df = df.map(remove_illegal_characters)

        return df

    else:
        raise ValueError("Invalid input format. Provide a string, list, or DataFrame.")


# Function to load and preprocess multiple data sources
def load_and_preprocess_multiple(data_sources, output_xls=r'C:\ws2024-principles-of-ai-engineering\preprocessed_combined.xlsx'):
    # List to store processed DataFrames
    dataframes = []

    # Define a mapping for renaming columns
    column_mappings = {
        "sample1.csv.gz": {
            "issue_label": "issue_label",
            "issue_title": "issue_title",
            "issue_body": "issue_body",
            "issue_created_at": "issue_created_at"
        },
        "sample2.csv.gz": {
            "issue_label": "issue_label",
            "issue_title": "issue_title",
            "issue_body": "issue_body",
            "issue_created_at": "issue_created_at"
        },
        "predictions.csv": {
            "title": "issue_title",
            "body": "issue_body",
            "corrected_label": "issue_label",
            "timestamp": "issue_created_at"
        }
    }

    # Load and normalize each data source
    for source in data_sources:
        if source.endswith('.gz'):
            with gzip.open(source, 'rt', encoding='ISO-8859-1') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(source, encoding='ISO-8859-1')

        # Get the filename to apply the appropriate column mapping
        filename = source.split("\\")[-1]  # Adjust path separator as needed
        if filename in column_mappings:
            df.rename(columns=column_mappings[filename], inplace=True)

        # Add missing columns to ensure consistent structure
        required_columns = [
            "issue_url", "issue_label", "issue_created_at", "issue_author_association",
            "repository_url", "issue_title", "issue_body"
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = None  # Fill missing columns with None

        # Add to the list of DataFrames
        dataframes.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Ensure the final DataFrame has only the required columns in the correct order
    final_columns = [
        "issue_url", "issue_label", "issue_created_at", "issue_author_association",
        "repository_url", "issue_title", "issue_body"
    ]
    combined_df = combined_df[final_columns]

    # Preprocess the combined data
    preprocessed_df = preprocess_text(combined_df)

    # Save the preprocessed data
    preprocessed_df.to_excel(output_xls, index=False, engine='openpyxl')
    print(f"Preprocessed data saved to {output_xls}")

    # Count records per category
    count_records_by_category(preprocessed_df)


# Entry point for standalone usage
if __name__ == '__main__':
    data_sources = [
        r'C:\ws2024-principles-of-ai-engineering\datasets\sample1.csv.gz',
        r'C:\ws2024-principles-of-ai-engineering\datasets\sample2.csv.gz',
        r'C:\ws2024-principles-of-ai-engineering\datasets\predictions.csv'
    ]
    output_file = r'C:\ws2024-principles-of-ai-engineering\preprocessed.xlsx'
    load_and_preprocess_multiple(data_sources, output_file)

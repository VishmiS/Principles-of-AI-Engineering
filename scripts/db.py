import sqlite3

# Define the database name
DB_NAME = 'predictions.db'


def init_db():
    """Initialize the database with required tables."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        # Create the unified predictions table
        # noinspection SqlDialectInspection,SqlNoDataSourceInspection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,                  -- Unique ID for each prediction
                title TEXT,                           -- Issue title
                body TEXT,                            -- Issue body/description
                predicted_label TEXT,                 -- Predicted label (e.g., Bug, Enhancement, Question)
                confidence REAL,                      -- Prediction confidence score
                corrected_label TEXT,                 -- Corrected label (if any)
                is_correct TEXT,                      -- Whether the prediction was correct
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP  -- Timestamp of the prediction or correction
            )
        ''')

        # Commit changes
        conn.commit()
        print("Database initialized successfully.")


def get_db_connection():
    """Create and return a new database connection."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Allows fetching rows as dictionaries
    return conn

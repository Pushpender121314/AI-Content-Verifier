import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    print("📥 Loading AI vs. Human Text dataset...")
    try:
        # Assumes a CSV structure with columns: 'text' and 'generated' (0=Human, 1=AI)
        df = pd.read_csv("AI_Human.csv")
    except FileNotFoundError:
        print("❌ Error: Place your dataset at 'AI_Human.csv' first.")
        return

    # Drop any null rows to prevent vectorization errors
    df = df.dropna().reset_index(drop=True)
    
    # Optional: If the dataset is massive (e.g., 500k rows), downsample to 40k rows 
    # to train in seconds on your i7 CPU while maintaining perfect accuracy
    if len(df) > 40000:
        df = df.sample(40000, random_state=42).reset_index(drop=True)

    X = df['text']
    y = df['generated']

    # 1. Initialize TF-IDF Vectorizer using word pairs (N-grams) to catch stylistic phrasing
    print("⚙️ Extracting structural text features using TF-IDF (N-grams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    # 2. Split dataset (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Train the Text Forensic Classifier
    print("🛡️ Training Generative AI Text Discriminator...")
    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_train, y_train)

    # 4. Evaluation Verification
    preds = model.predict(X_test)
    print("\n📊 Model Performance Matrix:")
    print(classification_report(y_test, preds))

    # 5. Export weights natively to this distinct folder
    joblib.dump(model, "ai_detector_model.pkl")
    joblib.dump(vectorizer, "text_vectorizer.pkl")
    print("✅ Training complete. Standalone text forensic assets generated successfully!")

if __name__ == "__main__":
    main()
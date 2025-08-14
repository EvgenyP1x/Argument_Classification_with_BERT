""" 2025 """

""" Argument Classification: Bag-of-Words pipeline"""


from .finetune import set_all_seeds

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from pathlib import Path
import pandas as pd
from fire import Fire


def bow_train():
    """
    Trains and evaluates baseline classifiers using bag-of-words (BoW) 
    Workflow:
        - load and split the dataset
        - apply multiple vectorizers
        - apply multiple classifiers
        - print accuracy for each vectorizer-classifier pair
    """
    set_all_seeds(555)

    dataset_dir = Path(__file__).parent / "Dataset"
    filepath = dataset_dir / 'reb_ref_dataset.xlsx'
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    dataset = pd.read_excel(filepath)

    X = dataset["Data"].str.strip()
    y = dataset["Label"].str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=555, stratify=y
    )
 
    # Vectorization methods
    vectorizers = {
        "CountVectorizer": CountVectorizer(),
        "TfidfVectorizer": TfidfVectorizer(),
        "CharNGramVectorizer": CountVectorizer(analyzer='char', ngram_range=(2,5)),
        "HashingVectorizer": HashingVectorizer(n_features=2**20)  # Optional, needs import
    }

    # Classifiers
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(max_iter=100000),
        "MultinomialNB": MultinomialNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "SGDClassifier": SGDClassifier(max_iter=1000, tol=1e-3),
    }

    # Evaluate each combination vectorizer + classifier
    for vec_name, vectorizer in vectorizers.items():
        for clf_name, classifier in classifiers.items():

            # Skip
            if clf_name == "MultinomialNB" and vec_name == "HashingVectorizer":
                print(f"Skipping {vec_name} + {clf_name} due to incompatibility.")
                continue

            try:
                if vec_name == "HashingVectorizer":
                    X_train_vec = vectorizer.transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)
                    # RandomForest and some classifiers require dense input
                    if clf_name in ["RandomForest", "SGDClassifier"]:
                        X_train_vec = X_train_vec.toarray()
                        X_test_vec = X_test_vec.toarray()
                    classifier.fit(X_train_vec, y_train)
                    acc = classifier.score(X_test_vec, y_test)

                else:
                    if clf_name in ["RandomForest"]:
                        X_train_vec = vectorizer.fit_transform(X_train)
                        X_test_vec = vectorizer.transform(X_test)
                        X_train_vec = X_train_vec.toarray()
                        X_test_vec = X_test_vec.toarray()
                        classifier.fit(X_train_vec, y_train)
                        acc = classifier.score(X_test_vec, y_test)
                    else:
                        model = make_pipeline(vectorizer, classifier)
                        model.fit(X_train, y_train)
                        acc = model.score(X_test, y_test)

                print(f"{vec_name} + {clf_name} accuracy: {acc:.4f}")

            except Exception as e:
                print(f"Failed {vec_name} + {clf_name} with error: {e}")


if __name__ == "__main__":

    Fire({"baseline_train": bow_train})
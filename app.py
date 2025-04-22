import pandas as pd
import numpy as np
import re
import nltk
import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE

# Suppress FutureWarning from imblearn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")

# Download NLTK data (run once during setup)
# nltk.download(['punkt', 'vader_lexicon', 'stopwords'])

# Clean and preprocess text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text.lower().strip()
    return ''

# Detect anxiety with balanced sentiment threshold
def detect_anxiety(text):
    keywords = ['anxious', 'worry', 'afraid', 'fear', 'panic', 'stress', 'scared', 'nervous', 'nightmare',
                'empty', 'disconnected', 'heavy', 'sad', 'alone', 'trapped', 'invisible', 'lost']
    keyword_score = 1 if isinstance(text, str) and any(k in text.lower() for k in keywords) else 0
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    implicit_score = 1 if sentiment['neg'] > 0.15 and sentiment['pos'] < 0.15 else 0
    return 1 if keyword_score or implicit_score else 0

# Extract thematic features with fixed regex
def extract_themes(text):
    if not isinstance(text, str):
        return {'falling': 0, 'flying': 0, 'chase': 0, 'test': 0, 'isolation': 0, 'loss': 0, 'darkness': 0, 'helplessness': 0}
    themes = {
        'falling': 1 if re.search(r'\b(fall(ing)?|fell)\b', text.lower()) else 0,
        'flying': 1 if re.search(r'\b(fly(ing)?|flew)\b', text.lower()) else 0,
        'chase': 1 if re.search(r'\b(chas(e|ing|ed))\b', text.lower()) else 0,
        'test': 1 if re.search(r'\b(test|exam)\b', text.lower()) else 0,
        'isolation': 1 if re.search(r'\b(alone|isolated|lonely|disconnected)\b', text.lower()) else 0,
        'loss': 1 if re.search(r'\b(lost|loss|gone)\b', text.lower()) else 0,
        'darkness': 1 if re.search(r'\b(dark|darkness|shadows?)\b', text.lower()) else 0,
        'helplessness': 1 if re.search(r'\b(helpless|powerless|stuck)\b', text.lower()) else 0
    }
    return themes

# Process dataset with additional features
def process_data(df, text_column='Text'):
    df['clean_text'] = df[text_column].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0].copy()
    df['anxiety_indicator'] = df['clean_text'].apply(detect_anxiety)
    
    # Add sentiment scores
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['clean_text'].apply(lambda x: sia.polarity_scores(x))
    df['negative_score'] = df['sentiment'].apply(lambda x: x['neg'])
    df['positive_score'] = df['sentiment'].apply(lambda x: x['pos'])
    df['composite_sentiment'] = df['negative_score'] - df['positive_score']
    
    # Add thematic features
    df['themes'] = df['clean_text'].apply(extract_themes)
    for theme in ['falling', 'flying', 'chase', 'test', 'isolation', 'loss', 'darkness', 'helplessness']:
        df[f'theme_{theme}'] = df['themes'].apply(lambda x: x[theme])
    
    # Print class distribution
    class_dist = df['anxiety_indicator'].value_counts().to_dict()
    print(f"Class Distribution: Class 0 (No Anxiety): {class_dist.get(0, 0)}, Class 1 (Anxiety): {class_dist.get(1, 0)}")
    
    return df

# Process single input text
def process_input_text(text, tfidf=None, vocab=None, max_length=200):
    clean_text_input = clean_text(text)
    anxiety = detect_anxiety(clean_text_input)
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(clean_text_input)
    negative_score = sentiment['neg']
    positive_score = sentiment['pos']
    composite_sentiment = negative_score - positive_score
    themes = extract_themes(clean_text_input)
    
    if tfidf:  # For ML models
        tfidf_vector = tfidf.transform([clean_text_input])
        extra_features = np.array([[negative_score, positive_score, composite_sentiment,
                                   themes['falling'], themes['flying'], themes['chase'], 
                                   themes['test'], themes['isolation'], themes['loss'], 
                                   themes['darkness'], themes['helplessness']]])
        features = hstack([tfidf_vector, extra_features * 2])  # Scale thematic features
        return features, anxiety, extra_features, sentiment, themes
    
    elif vocab:  # For LSTM model
        tokens = word_tokenize(clean_text_input)
        indices = [vocab[token] for token in tokens]
        text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)[:, :max_length]
        padded = torch.zeros(1, max_length, dtype=torch.long)
        padded[:, :min(len(indices), max_length)] = text_tensor
        anxiety_tensor = torch.tensor([anxiety], dtype=torch.float).unsqueeze(1)
        return padded, anxiety_tensor, None, sentiment, themes
    
    return None, None, None, sentiment, themes

# Prepare features and split data
def prepare_features(df, target_column='anxiety_indicator', model_type='ml', max_words=10000):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train = train_df[target_column].values
    y_test = test_df[target_column].values
    
    if model_type == 'ml':
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = tfidf.fit_transform(train_df['clean_text'])
        X_test_tfidf = tfidf.transform(test_df['clean_text'])
        
        # Add thematic and sentiment features
        extra_features = ['negative_score', 'positive_score', 'composite_sentiment',
                         'theme_falling', 'theme_flying', 'theme_chase', 'theme_test',
                         'theme_isolation', 'theme_loss', 'theme_darkness', 'theme_helplessness']
        X_train_extra = train_df[extra_features].values
        X_test_extra = test_df[extra_features].values
        
        # Scale thematic features
        X_train_extra[:, 3:] *= 2  # Scale theme features (after sentiment scores)
        X_test_extra[:, 3:] *= 2
        
        # Combine TF-IDF and extra features
        X_train = hstack([X_train_tfidf, X_train_extra])
        X_test = hstack([X_test_tfidf, X_test_extra])
        
        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'feature_extractor': tfidf}
    
    elif model_type == 'lstm':
        from torchtext.vocab import build_vocab_from_iterator
        from torch.nn.utils.rnn import pad_sequence
        max_length = 200
        def yield_tokens(texts):
            for text in texts:
                yield word_tokenize(text)
        vocab = build_vocab_from_iterator(yield_tokens(train_df['clean_text']), 
                                         max_tokens=max_words, specials=['<pad>', '<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        
        X_train = [torch.tensor([vocab[token] for token in word_tokenize(text)], dtype=torch.long) 
                   for text in train_df['clean_text']]
        X_test = [torch.tensor([vocab[token] for token in word_tokenize(text)], dtype=torch.long) 
                  for text in test_df['clean_text']]
        X_train = pad_sequence(X_train, batch_first=True, padding_value=vocab['<pad>'])[:, :max_length]
        X_test = pad_sequence(X_test, batch_first=True, padding_value=vocab['<pad>'])[:, :max_length]
        
        # Add anxiety indicator as extra feature
        X_train_anxiety = torch.tensor(train_df['anxiety_indicator'].values, dtype=torch.float).unsqueeze(1)
        X_test_anxiety = torch.tensor(test_df['anxiety_indicator'].values, dtype=torch.float).unsqueeze(1)
        
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 
                'X_train_anxiety': X_train_anxiety, 'X_test_anxiety': X_test_anxiety,
                'feature_extractor': vocab, 'vocab_size': len(vocab), 'max_length': max_length}

# LSTM model with extra feature and dropout
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2 + 1, 1)  # +1 for anxiety feature
    
    def forward(self, text, anxiety_feature):
        embedded = self.embedding(text)
        output, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        combined = torch.cat((hidden, anxiety_feature), dim=1)
        return self.fc(combined)  # Output logits

# Train LSTM model with early stopping
def train_lstm(model, X_train, X_train_anxiety, y_train, epochs=10, batch_size=32, patience=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train = X_train.to(device)
    X_train_anxiety = X_train_anxiety.to(device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_train, X_train_anxiety, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_anxiety, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X, batch_anxiety).view(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

# Plot confusion matrix, ROC curve, precision-recall curve, and feature importance
def plot_metrics(y_test, y_pred, y_pred_proba, model_name, output_dir, model=None, tfidf=None, extra_feature_names=None):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_pr_curve.png'))
    plt.close()
    
    # Feature Importance (for Random Forest)
    if model_name == 'random_forest' and model is not None and extra_feature_names is not None and tfidf is not None:
        importances = model.feature_importances_
        n_tfidf = tfidf.get_feature_names_out().shape[0]
        extra_indices = np.arange(n_tfidf, n_tfidf + len(extra_feature_names))
        extra_importances = importances[extra_indices]
        top_k = 10
        sorted_idx = np.argsort(extra_importances)[-top_k:][::-1]
        top_feature_labels = [extra_feature_names[i] for i in sorted_idx]
        top_feature_values = extra_importances[sorted_idx]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(top_feature_values)), top_feature_values, align='center')
        plt.yticks(range(len(top_feature_labels)), top_feature_labels)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance (Extra Features) - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))
        plt.close()

# Evaluate models
def evaluate_models(models, X_test, y_test, model_type='ml', X_test_anxiety=None, output_dir='results_dir', tfidf=None, extra_feature_names=None):
    results = {}
    y_test_np = y_test
    
    if model_type == 'ml':
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            results[name] = {
                'accuracy': model.score(X_test, y_test_np),
                'roc_auc': roc_auc_score(y_test_np, y_pred_proba),
                'classification_report': classification_report(y_test_np, y_pred, output_dict=True, zero_division=0)
            }
            plot_metrics(y_test_np, y_pred, y_pred_proba, name, output_dir, model, tfidf, extra_feature_names)
    
    elif model_type == 'lstm':
        model = models['lstm']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            X_test_anxiety = X_test_anxiety.to(device)
            y_pred_proba = torch.sigmoid(model(X_test, X_test_anxiety)).squeeze().cpu().numpy()
            y_pred = (y_pred_proba > 0.5).astype(int)
            results['lstm'] = {
                'accuracy': (y_pred == y_test_np).mean(),
                'roc_auc': roc_auc_score(y_test_np, y_pred_proba),
                'classification_report': classification_report(y_test_np, y_pred, output_dict=True, zero_division=0)
            }
            plot_metrics(y_test_np, y_pred, y_pred_proba, 'lstm', output_dir)
    
    # Save results to CSV
    results_df = []
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'ROC_AUC': metrics['roc_auc'],
            'Class_0_Precision': metrics['classification_report']['0']['precision'],
            'Class_0_Recall': metrics['classification_report']['0']['recall'],
            'Class_0_F1': metrics['classification_report']['0']['f1-score'],
            'Class_1_Precision': metrics['classification_report']['1']['precision'],
            'Class_1_Recall': metrics['classification_report']['1']['recall'],
            'Class_1_F1': metrics['classification_report']['1']['f1-score']
        }
        results_df.append(row)
    
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    
    # Print summary table
    print("\nSummary of Model Performance:")
    print(f"{'Model':<15} {'Accuracy':<10} {'ROC AUC':<10} {'Class 0 Precision':<18} {'Class 0 Recall':<15} {'Class 0 F1':<12} {'Class 1 Precision':<18} {'Class 1 Recall':<15} {'Class 1 F1':<12}")
    print("-" * 120)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<15} {row['Accuracy']:<10.4f} {row['ROC_AUC']:<10.4f} {row['Class_0_Precision']:<18.4f} {row['Class_0_Recall']:<15.4f} {row['Class_0_F1']:<12.4f} {row['Class_1_Precision']:<18.4f} {row['Class_1_Recall']:<15.4f} {row['Class_1_F1']:<12.4f}")
    
    # Print detailed results
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print("Classification Report:")
        for class_name, values in metrics['classification_report'].items():
            if isinstance(values, dict) and 'precision' in values:
                print(f"  Class {class_name}:")
                print(f"    Precision: {values['precision']:.4f}")
                print(f"    Recall: {values['recall']:.4f}")
                print(f"    F1-Score: {values['f1-score']:.4f}")
            else:
                print(f"  {class_name}: {values}")

    return results

# Predict anxiety for a single input
def predict_anxiety(models, model_type, feature_extractor, output_dir, max_length=200):
    extra_feature_names = ['negative_score', 'positive_score', 'composite_sentiment',
                          'theme_falling', 'theme_flying', 'theme_chase', 'theme_test',
                          'theme_isolation', 'theme_loss', 'theme_darkness', 'theme_helplessness']
    print("\nEnter a dream description (or 'quit' to exit):")
    while True:
        text = input("> ")
        if text.lower() == 'quit':
            break
        if not text.strip():
            print("Please enter a valid dream description.")
            continue
        
        if model_type == 'ml':
            features, _, extra_features, sentiment, themes = process_input_text(text, tfidf=feature_extractor)
            for name, model in models.items():
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]
                label = "Anxiety" if pred == 1 else "No Anxiety"
                print(f"\n{name} Prediction: {label} (Probability of Anxiety: {prob:.4f})")
                print("Input Analysis:")
                print(f"  Sentiment Scores: Negative={sentiment['neg']:.4f}, Positive={sentiment['pos']:.4f}, Composite={sentiment['neg'] - sentiment['pos']:.4f}")
                print("  Detected Themes:", [k for k, v in themes.items() if v == 1] or "None")
                if name == 'random_forest' and pred == 1:
                    importances = model.feature_importances_
                    n_tfidf = feature_extractor.get_feature_names_out().shape[0]
                    extra_indices = np.arange(n_tfidf, n_tfidf + len(extra_feature_names))
                    extra_importances = importances[extra_indices]
                    indices = np.argsort(extra_importances)[-3:][::-1]
                    print("Key Features Influencing Anxiety Prediction:")
                    for i in indices:
                        print(f"  - {extra_feature_names[i]}: {extra_importances[i]:.4f}")
        
        elif model_type == 'lstm':
            model = models['lstm']
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.eval()
            features, anxiety_feature, _, sentiment, themes = process_input_text(text, vocab=feature_extractor, max_length=max_length)
            with torch.no_grad():
                features = features.to(device)
                anxiety_feature = anxiety_feature.to(device)
                prob = torch.sigmoid(model(features, anxiety_feature)).squeeze().cpu().numpy()
                pred = 1 if prob > 0.5 else 0
                label = "Anxiety" if pred == 1 else "No Anxiety"
                print(f"\nLSTM Prediction: {label} (Probability of Anxiety: {prob:.4f})")
                print("Input Analysis:")
                print(f"  Sentiment Scores: Negative={sentiment['neg']:.4f}, Positive={sentiment['pos']:.4f}, Composite={sentiment['neg'] - sentiment['pos']:.4f}")
                print("  Detected Themes:", [k for k, v in themes.items() if v == 1] or "None")
                if pred == 1 and anxiety_feature.item() == 1:
                    print("Key Feature Influencing Anxiety Prediction: Anxiety Indicator Present")
        
        # Save prediction to CSV
        pred_df = pd.DataFrame([{
            'Input_Text': text,
            'Model': name if model_type == 'ml' else 'lstm',
            'Prediction': label,
            'Anxiety_Probability': prob,
            'Negative_Sentiment': sentiment['neg'],
            'Positive_Sentiment': sentiment['pos'],
            'Detected_Themes': str([k for k, v in themes.items() if v == 1])
        }])
        pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), mode='a', index=False, header=not os.path.exists(os.path.join(output_dir, 'predictions.csv')))

# Main function
def main():
    parser = argparse.ArgumentParser(description='Dream Analysis System')
    parser.add_argument('--data', type=str, default='dream_dataset.csv', help='Path to dataset CSV')
    parser.add_argument('--output', type=str, default='results_dir', help='Output directory')
    parser.add_argument('--model', type=str, choices=['ml', 'lstm'], default='ml', help='Model type')
    args = parser.parse_args()

    # Verify data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and process data
    df = pd.read_csv(args.data)
    df = process_data(df)
    
    # Extra feature names for feature importance
    extra_feature_names = ['negative_score', 'positive_score', 'composite_sentiment',
                          'theme_falling', 'theme_flying', 'theme_chase', 'theme_test',
                          'theme_isolation', 'theme_loss', 'theme_darkness', 'theme_helplessness']
    
    if args.model == 'ml':
        # Train and evaluate traditional ML models
        ml_data = prepare_features(df, model_type='ml')
        ml_models = {
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight={0: 1, 1: 3}),
            'neural_network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=2000, random_state=42, alpha=0.001, early_stopping=True, validation_fraction=0.2)
        }
        # Optimize Random Forest with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 3}), param_grid, cv=5, scoring='recall_macro')
        grid.fit(ml_data['X_train'], ml_data['y_train'])
        ml_models['random_forest'] = grid.best_estimator_
        
        for name, model in ml_models.items():
            if name == 'random_forest':
                continue  # Already fitted via GridSearchCV
            model.fit(ml_data['X_train'], ml_data['y_train'])  # Removed sample_weight
            print(f"{name} Training Loss: {model.loss_:.4f}")  # Debug training loss
        print("Traditional ML Results:")
        evaluate_models(ml_models, ml_data['X_test'], ml_data['y_test'], model_type='ml', 
                        output_dir=args.output, tfidf=ml_data['feature_extractor'], extra_feature_names=extra_feature_names)
        predict_anxiety(ml_models, 'ml', ml_data['feature_extractor'], args.output)
    
    elif args.model == 'lstm':
        # Train and evaluate LSTM model
        lstm_data = prepare_features(df, model_type='lstm')
        lstm_model = LSTMModel(vocab_size=lstm_data['vocab_size'])
        lstm_model = train_lstm(lstm_model, lstm_data['X_train'], lstm_data['X_train_anxiety'], lstm_data['y_train'])
        print("LSTM Results:")
        evaluate_models({'lstm': lstm_model}, lstm_data['X_test'], lstm_data['y_test'], 
                        model_type='lstm', X_test_anxiety=lstm_data['X_test_anxiety'], output_dir=args.output)
        predict_anxiety({'lstm': lstm_model}, 'lstm', lstm_data['feature_extractor'], args.output, lstm_data['max_length'])
if __name__ == "__main__":
    main()
from flask import Flask, request, jsonify
import pandas as pd
import io 
import numpy as np 

app = Flask(__name__)
# Attribute to hold the DataFrame
app.dataset = None  # Initialize the attribute to None

# Route to upload file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Determine the file format
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.json'):
        df = pd.read_json(file)
    elif file.filename.endswith('.parquet'): 
        df=pd.read_parquet(file)
    else:
        return jsonify({'message': 'Unsupported file format'}), 400

    # Save dataset for later EDA
    app.dataset = df  # Store DataFrame in app attribute
    return jsonify({'message': 'File upload succeeded', 'size': df.shape}), 200

# Route to perform EDA
@app.route('/eda', methods=['GET'])
def perform_eda():
  if app.dataset is None:
        return jsonify({'message': 'No dataset uploaded for EDA'}), 400

  df = app.dataset  # Retrieve the DataFrame from the app attribute

    # Buffer for df.info() output
  buffer = io.StringIO()
  df.info(buf=buffer)
  info_str = buffer.getvalue()

    # Basic EDA information
  data_info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'columns_list': df.columns.tolist(),
        'info': info_str
    }
  return jsonify(data_info), 200
# Route to count the number of languages
@app.route('/languages_count', methods=['GET'])
def languages_count():
    df = app.dataset
    if df is None:
        return jsonify({'message': 'Dataset not found'}), 400

    if 'lang' in df.columns:
        languages_count = df['lang'].value_counts().to_dict()
        return jsonify({'languages_count': languages_count}), 200
    else:
        return jsonify({'message': 'No language column found in the dataset'}), 400
# Manually map the fixed labels to the five categories
label_to_category_mapping = {
    'helpfulness': 'Paraphrasing',
    'creativity': 'Summarization',
    'lang_mismatch': 'Translation',
    'quality': 'Summarization',
    'spam': 'Miscellaneous',
    'fails_task': 'Miscellaneous',
    'pii': 'Miscellaneous',
    'not_appropriate': 'Miscellaneous',
    'hate_speech': 'Miscellaneous',
    'sexual_content': 'Miscellaneous',
    'toxicity': 'Miscellaneous',
    'humor': 'Role-play',
    'violence': 'Miscellaneous'
}

# Function to classify each message based on the highest value in 'labels'
def categorize_message(label_data):
    if label_data is None or not isinstance(label_data, dict) or 'name' not in label_data or 'value' not in label_data:
        return 'Miscellaneous'

    categories = label_data['name']
    values = label_data['value']

    if len(values) == 0 or len(categories) == 0:
        return 'Miscellaneous'

    max_value_index = np.argmax(values)
    selected_label = categories[max_value_index]

    return label_to_category_mapping.get(selected_label, 'Miscellaneous')

# Route to perform user intent analysis
@app.route('/userintents', methods=['GET'])
def user_intents():
    try:
        # Load dataset
        df = app.dataset # Adjust the path to your dataset
        print("Loaded DataFrame:")
        print(df.head())  # Print the first few rows to debug

    except FileNotFoundError:
        return jsonify({'message': 'Dataset not found'}), 400

    # Check if 'labels' column exists
    if 'labels' not in df.columns:
        return jsonify({'message': "'labels' column is missing in the dataset"}), 400

    # Apply the classification function to all rows in the dataset
    df['category'] = df['labels'].apply(categorize_message)

    # Count the number of messages that fall into each of the five main categories
    category_counts = df['category'].value_counts().to_dict()
    
    print("Categorized counts:")
    print(category_counts)  # Print categorized counts for debugging

    # Return the counts as a JSON response
    return jsonify({'user_intents_count': category_counts}), 200 

@app.route('/toxicity', methods=['GET'])
def get_toxicity_scores():
    if app.dataset is None:
        return jsonify({'message': 'No dataset uploaded for analysis'}), 400

    df = app.dataset

    if 'detoxify' not in df.columns:
        return jsonify({'message': "No 'detoxify' column found in the dataset"}), 400

    # Extract toxicity scores and handle None values
    toxicity_scores = df["detoxify"].apply(lambda x: x['toxicity'] if x is not None else None)
    toxicity_scores = toxicity_scores.dropna()

    return jsonify({'toxicity_scores': toxicity_scores.tolist()}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

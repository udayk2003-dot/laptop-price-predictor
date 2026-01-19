from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
import requests

app = Flask(__name__)

# Load dataset with tab separator
df = pd.read_csv('dataset1.csv')


# Load model and feature order columns
model = pickle.load(open('rf_model.pkl', 'rb'))
feature_order = pickle.load(open('columns.pkl', 'rb'))

# Prepare dropdown lists from dataset unique values
brands = sorted(df['brand'].dropna().unique())
processors = sorted(df['processor'].dropna().unique())
gpus = sorted(df['GPU'].dropna().unique())
ram_types = sorted(df['Ram_type'].dropna().unique())
rom_types = sorted(df['ROM_type'].dropna().unique())
display_sizes = sorted(df['display_size'].dropna().unique())

# Combine resolution columns
df['resolution'] = df['resolution_width'].astype(str) + 'x' + df['resolution_height'].astype(str)
resolutions = sorted(df['resolution'].dropna().unique())

# Remove 0 from warranty values
warranties = sorted(w for w in df['warranty'].dropna().unique() if w != 0)

# Load comparison dataset separately
comparison_df = pd.read_csv(
    'comparison_dataset.csv',
    sep='\t',
    engine='python',
    on_bad_lines='skip'
)



# Your NewsAPI key
NEWS_API_KEY = "478dd1f94f344c3abec1617720b9a6ac"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/price-prediction')
def price_prediction():
    return render_template('prediction.html',
                           brands=brands,
                           processors=processors,
                           gpus=gpus,
                           ram_types=ram_types,
                           rom_types=rom_types,
                           display_sizes=display_sizes,
                           resolutions=resolutions,
                           warranties=warranties)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        brand = request.form['brand']
        processor = request.form['processor']
        gpu = request.form['gpu']
        ram_type = request.form['ram_type']
        ram = int(request.form['ram'])
        rom_type = request.form['rom_type']
        rom = int(request.form['rom'])
        display_size = float(request.form['display_size'])
        resolution = request.form['resolution']
        warranty = int(request.form['warranty'])

        resolution_width, resolution_height = resolution.split('x')
        resolution_width = int(resolution_width)
        resolution_height = int(resolution_height)

        input_dict = {
            'brand': brand,
            'processor': processor,
            'GPU': gpu,
            'Ram_type': ram_type,
            'Ram': ram,
            'ROM_type': rom_type,
            'ROM': rom,
            'display_size': display_size,
            'resolution_width': resolution_width,
            'resolution_height': resolution_height,
            'warranty': warranty
        }

        input_df = pd.DataFrame(columns=feature_order)
        for col in feature_order:
            input_df.at[0, col] = input_dict.get(col, 0)

        predicted_price = model.predict(input_df)[0]
        predicted_price = round(predicted_price)

        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Error during prediction. Please check your inputs.'})

# --- Comparison Routes ---

@app.route('/get_laptops')
def get_laptops():
    laptops = sorted(comparison_df['Model'].dropna().unique())
    return jsonify(laptops)

@app.route('/get_specs', methods=['POST'])
def get_specs():
    data = request.get_json()
    model_name = data.get('model')
    if not model_name:
        return jsonify({})

    laptop_df = comparison_df[comparison_df['Model'] == model_name]
    if laptop_df.empty:
        return jsonify({})

    laptop = laptop_df.iloc[0]

    def safe_get(col):
        val = laptop.get(col, None)
        if pd.isna(val):
            return None
        if hasattr(val, 'item'):
            return val.item()
        return val

    specs = {
        'Brand': safe_get('brand'),
        'Price': safe_get('Price'),
        'Display': safe_get('display_size'),
        'Processor': f"{safe_get('processor_brand') or ''} {safe_get('processor_tier') or ''}".strip() or None,
        'GPU': f"{safe_get('gpu_brand') or ''} {safe_get('gpu_type') or ''}".strip() or None,
        'Resolution Width': safe_get('resolution_width'),
        'Resolution Height': safe_get('resolution_height'),
        'Warranty': safe_get('year_of_warranty'),
        'RAM': safe_get('ram_memory'),
        'ROM': f"{safe_get('primary_storage_capacity') or ''} {safe_get('primary_storage_type') or ''}".strip() or None,
    }

    return jsonify(specs)


# --- UPDATED NEWS ROUTE with category filtering ---

@app.route('/news')
def news():
    category = request.args.get('category', 'all').lower()

    CATEGORY_QUERIES = {
        "all": "",  # general technology news
        "product launch": "product launch",
        "hardware": "hardware",
        "review": "review",
        "industry insight": "industry insight",
        "buying guide": "buying guide",
        "tech trends": "tech trends"
    }

    query = CATEGORY_QUERIES.get(category, "")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query if query else "technology",
        "language": "en",
        "pageSize": 9,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        articles = data.get('articles', [])
    except Exception as e:
        print("NewsAPI error:", e)
        articles = []

    return render_template('news.html', articles=articles, current_category=category.title())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

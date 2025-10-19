import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib # for saving the model and vectorizer

# --- 1. Constants and File Setup ---
FILE_PATH = "Salary_Dataset_with_Extra_Features.csv"
RANDOM_STATE = 42
TARGET_COL = 'Salary'

# --- 2. Data Loading and Cleaning Function ---

# Simple text cleaning function for Job Title
def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower() 
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads, cleans, and selects the required columns."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None, None

    # Filter columns as requested (keeping Salary as the target)
    COLUMNS_TO_KEEP = [
        'Company Name', 
        'Job Title', 
        'Location', 
        'Employment Status', 
        'Job Roles', 
        TARGET_COL
    ]
    df = df[COLUMNS_TO_KEEP].copy()

    # Filter out intern positions and get clean job titles
    df_filtered = df[~df['Job Title'].str.contains('Intern|intern', case=False, na=False)]
    
    # Get unique job titles and clean them (remove extra details)
    unique_job_titles = []
    for title in df_filtered['Job Title'].unique():
        # Remove common suffixes/prefixes to simplify
        clean_title = title
        # Remove contractor, consultant, etc.
        clean_title = re.sub(r'\s*-\s*(Contractor|Contract|Consultant).*$', '', clean_title, flags=re.IGNORECASE)
        clean_title = re.sub(r'\s*\(.*?\)', '', clean_title)  # Remove parentheses content
        clean_title = clean_title.strip()
        if clean_title and clean_title not in unique_job_titles:
            unique_job_titles.append(clean_title)
    
    unique_job_titles = sorted(unique_job_titles)

    # Preprocess the Target Variable (Log transformation for better regression)
    df['log_Salary'] = np.log1p(df[TARGET_COL])
    
    df['Job Title'] = df['Job Title'].apply(preprocess_text)
    
    return df, unique_job_titles

# --- 3. Model Training Function ---

@st.cache_resource
def train_model(df):
    """Trains the Ridge Regression model using a ColumnTransformer/Pipeline."""
    
    # Define features and target
    X = df.drop([TARGET_COL, 'log_Salary'], axis=1)
    y = df['log_Salary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 3.1. Define Preprocessing Steps
    
    # NLP for 'Job Title'
    text_features = 'Job Title'
    text_transformer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    
    # One-Hot Encoding for categorical features
    cat_features = ['Company Name', 'Location', 'Employment Status', 'Job Roles']
    cat_transformer = OneHotEncoder(handle_unknown='ignore') # ignore unknown categories in test/live data

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', text_transformer, text_features),
            ('cat_ohe', cat_transformer, cat_features)
        ],
        remainder='passthrough', # keep any other columns (none in this case)
        verbose_feature_names_out=False
    )
    
    # 3.2. Create and Train the Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])
    
    with st.spinner("Training Machine Learning Model..."):
        model_pipeline.fit(X_train, y_train)

    return model_pipeline, X_train, y_train, X_test, y_test

# --- 4. Prediction Function ---

def predict_salary(model, input_df):
    """Makes a prediction and converts it back to the original scale."""
    log_pred = model.predict(input_df)[0]
    
    # Inverse transform (expm1 is the inverse of log1p)
    predicted_salary = np.expm1(log_pred)
    
    # Ensure prediction is not negative
    return max(0, predicted_salary)


# ====================================================================
# --- 5. Streamlit Application UI ---
# ====================================================================

st.set_page_config(
    page_title="NLP Job Salary Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Load Data
result = load_and_preprocess_data(FILE_PATH)

if result[0] is not None:
    df, job_titles_list = result
    # Train Model
    pipeline, X_train, y_train, X_test, y_test = train_model(df)
    
    st.title("💰 NLP-Powered Job Salary Predictor")
    st.markdown(
        """
        This application predicts job salary based on **Job Title** (using NLP) 
        and other key features like **Company**, **Location**, and **Employment Status**.
        """
    )
    
    st.divider()

    # --- Sidebar for Data Exploration ---
    with st.sidebar:
        st.header("Dataset Overview")
        st.write(f"Total Records: **{len(df):,}**")
        st.dataframe(df.head(5), width="stretch")
        
        # Display sample statistics of the actual salary
        st.subheader("Salary Distribution (₹)")
        st.write(df[TARGET_COL].describe().apply(lambda x: f'{x:,.0f}'))

    # --- Main Area for User Input ---
    
    st.header("Input Job Details")
    
    # 5.1. Job Title Input (The main NLP feature)
    job_title = st.selectbox(
        "Job Title", 
        options=job_titles_list,
        index=job_titles_list.index('Data Scientist') if 'Data Scientist' in job_titles_list else 0,
        help="Select a job title from the dataset (intern positions excluded)"
    )

    col1, col2 = st.columns(2)
    
    # 5.2. Categorical Inputs
    with col1:
        # Use unique values from the dataset for dropdowns (limit to top 50 for company/location)
        top_companies = df['Company Name'].value_counts().nlargest(50).index.tolist()
        company = st.selectbox(
            "Company Name (Top 50)", 
            options=top_companies + ["Other"],
            index=0
        )
        
        locations = df['Location'].unique().tolist()
        location = st.selectbox(
            "Location", 
            options=locations,
            index=locations.index('Bangalore') if 'Bangalore' in locations else 0
        )
        
    with col2:
        employment_statuses = df['Employment Status'].unique().tolist()
        employment_status = st.selectbox(
            "Employment Status", 
            options=employment_statuses
        )
        
        job_roles = df['Job Roles'].unique().tolist()
        job_role = st.selectbox(
            "Job Role Category", 
            options=job_roles
        )

    # 5.3. Prediction Button
    if st.button("Predict Salary", type="primary", width="stretch"):
        
        # Create a DataFrame from user inputs for prediction
        input_data = pd.DataFrame([{
            'Company Name': company, 
            'Job Title': job_title, 
            'Location': location, 
            'Employment Status': employment_status,
            'Job Roles': job_role
        }])
        
        # Clean the input Job Title just like the training data
        input_data['Job Title'] = input_data['Job Title'].apply(preprocess_text)
        
        # Make Prediction
        predicted_salary = predict_salary(pipeline, input_data)
        
        st.success("---")
        st.metric(
            label="Predicted Annual Salary (INR)", 
            value=f"₹ {predicted_salary:,.0f}",
            delta=None # You can add a difference from average here if you calculate it
        )
        st.balloons()
        
        st.caption("Disclaimer: This is an estimate based on the provided dataset and machine learning model.")
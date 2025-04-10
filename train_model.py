import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def process_course(file_path):
    """Process individual course files"""
    df = pd.read_csv(file_path)
    course_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Clean and prepare data
    df = df.drop(columns=['Gender'], errors='ignore').drop_duplicates()
    df['text_features'] = df['Learning Strategy'] + ' ' + df['Additional Notes']
    
    return course_name, df

def train_model():
    """Train and save per-course recommendation models"""
    model_data = {}
    
    for file in os.listdir('./course_data'):
        if file.endswith('.csv'):
            course_path = os.path.join('./course_data', file)
            course_name, course_df = process_course(course_path)
            
            # Train TF-IDF and KNN for this course
            tfidf = TfidfVectorizer(stop_words='english')
            X = tfidf.fit_transform(course_df['text_features'])
            
            # Always get 2 neighbors (or max available)
            n_neighbors = min(2, len(course_df))
            knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
            knn.fit(X)
            
            # Store course data with original indices
            model_data[course_name] = {
                'tfidf': tfidf,
                'knn': knn,
                'metadata': course_df.to_dict('records')
            }
    
    joblib.dump(model_data, 'course_recommender.joblib')
    print(f'Trained models for {len(model_data)} courses!')

if __name__ == '__main__':
    train_model()
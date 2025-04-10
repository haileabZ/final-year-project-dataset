import joblib

def get_recommendations(course_name):
    """Get 2 top recommendations for a specific course"""
    try:
        model_data = joblib.load('course_recommender.joblib')
        
        if course_name not in model_data:
            available_courses = "\n- ".join(model_data.keys())
            return f"🚫 Course '{course_name}' not found! Available courses:\n- {available_courses}"
        
        course_model = model_data[course_name]
        tfidf = course_model['tfidf']
        knn = course_model['knn']
        metadata = course_model['metadata']
        
        _, indices = knn.kneighbors(tfidf.transform([metadata[0]['text_features']]))
        
        results = []
        seen = set()
        
        for idx in indices[0]:
            if len(results) >= 2:
                break
            entry = metadata[idx]
            unique_id = f"{entry['Recommended Channel/Link']}-{entry['Learning Strategy']}"
            
            if unique_id not in seen:
                seen.add(unique_id)
                results.append(
                    f"📺 Channel: {entry['Recommended Channel/Link']}\n"
                    f"📚 Strategy: {entry['Learning Strategy']}\n"
                    f"📝 Notes: {entry['Additional Notes']}\n"
                    f"⏳ Hours/Week: {entry['Study Hours Per Week']}\n"
                    f"⏰ Best Time: {entry['Preferred Study Time']}\n"
                )
                
        return "\n".join(results) if results else "❌ No recommendations available"
        
    except FileNotFoundError:
        return "❌ Model not found! Train first with train_model.py"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

if __name__ == "__main__":
    print("=== Course Recommendation System ===")
    print("Type 'exit' to quit\n")
    
    while True:
        course_name = input("Enter course name: ").strip()
        
        if course_name.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not course_name:
            print("Please enter a course name\n")
            continue
            
        print(f"\n🔍 Recommendations for {course_name}:")
        print(get_recommendations(course_name))
        print("\n" + "-"*50 + "\n")
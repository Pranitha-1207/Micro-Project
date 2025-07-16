# ==============================================================================
# SCRIPT: api.py (with Firebase Integration)
# ==============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore, auth
import datetime

try:
    cred = credentials.Certificate("firebase-credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase connected successfully.")
except Exception as e:
    print(f"Firebase connection error: {e}")
    db = None

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(title="Advanced Rent Prediction API", version="2.0")

# --- LOAD ML MODEL ---
model = joblib.load('house_rent_predictor.joblib')
model_columns = joblib.load('model_columns.joblib')

# --- PYDANTIC MODELS FOR DATA VALIDATION ---
class HouseData(BaseModel):
    BHK: int
    Size: int
    Bathroom: int
    City: str
    FurnishingStatus: str
    userId: str  # To link prediction to a user

class FeedbackData(BaseModel):
    userId: str
    predicted_rent: float
    actual_rent: float
    house_details: dict # Store the features that led to the prediction

class UserCredentials(BaseModel):
    email: str
    password: str



# --- HELPER PREDICTION FUNCTION ---
def make_prediction(data: HouseData):
    input_df = pd.DataFrame([data.dict(exclude={'userId'})]) # Exclude userId for prediction
    input_df = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_df)[0]
    return prediction

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Advanced Rent Prediction API!"}

# =========================================================
# REPLACE THE OLD predict_and_save FUNCTION IN api.py
# =========================================================

@app.post("/predict")
async def predict_and_save(data: HouseData):
    """
    Predicts rent and saves the prediction to the user's history in Firestore.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database service not available.")

    # --- THE FIX IS HERE ---
    # We explicitly convert the model's output (a numpy float) to a standard Python float.
    # Firestore can handle Python floats, but not numpy floats.
    prediction = float(make_prediction(data))
    
    # Now, we can safely save the prediction record to Firestore
    prediction_record = {
        "userId": data.userId,
        "prediction": prediction,  # This is now a standard float
        "house_details": data.dict(exclude={'userId'}),
        "timestamp": datetime.datetime.utcnow()
    }
    
    try:
        db.collection('predictions').add(prediction_record)
    except Exception as e:
        # If there's an error saving to Firestore, return an informative error
        raise HTTPException(status_code=500, detail=f"Error saving prediction to database: {e}")
    
    # Return the prediction to the frontend
    return {"predicted_rent": prediction}

@app.post("/submit_feedback")
async def submit_feedback(data: FeedbackData):
    """
    Receives user feedback (actual rent) and stores it for model improvement.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database service not available.")
        
    feedback_record = {
        "userId": data.userId,
        "predicted_rent": data.predicted_rent,
        "actual_rent": data.actual_rent,
        "house_details": data.house_details,
        "timestamp": datetime.datetime.utcnow()
    }
    db.collection('feedback').add(feedback_record)
    
    return {"status": "success", "message": "Thank you for your feedback!"}

# =========================================================
# REPLACE THE get_prediction_history FUNCTION IN api.py
# WITH THIS MORE ROBUST VERSION
# =========================================================

from google.cloud.firestore_v1.base_query import FieldFilter # <-- ADD THIS IMPORT AT THE TOP OF api.py

# ... (rest of your imports and code) ...

@app.get("/history/{user_id}")
async def get_prediction_history(user_id: str):
    """
    Fetches the prediction history for a given user from Firestore.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database service not available.")

    try:
        # The query logic remains the same.
        # Note: I'm replacing the older where() syntax with the newer FieldFilter for clarity,
        # but your existing .where('userId', '==', user_id) will also work fine.
        predictions_ref = db.collection('predictions').where(filter=FieldFilter("userId", "==", user_id)).order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        
        history_data = []
        for pred in predictions_ref:
            p_data = pred.to_dict()
            
            # --- ROBUST DATA TYPE CONVERSION ---
            # This block will now handle all potential data type issues before creating the JSON response.
            
            # 1. Handle the timestamp: Convert Firestore Timestamp or Python datetime to a string.
            if 'timestamp' in p_data:
                # The object from Firestore might not be a standard Python datetime,
                # so we call its .isoformat() method if it exists, or convert it.
                ts = p_data['timestamp']
                if hasattr(ts, 'isoformat'):
                    p_data['timestamp'] = ts.isoformat()
                else:
                    # Fallback for other potential types
                    p_data['timestamp'] = str(ts)

            # 2. Handle the prediction value (handles both numpy.float and standard float).
            if 'prediction' in p_data:
                p_data['prediction'] = float(p_data['prediction'])
            
            history_data.append(p_data)

        return {"history": history_data}
    
    except Exception as e:
        # Adding a print statement here will help you see the exact error in your API terminal
        print(f"Error in /history endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching history: {e}")
    
@app.post("/register")
async def register_user_api(credentials: UserCredentials):
    """
    Creates a new user in Firebase Authentication.
    """
    try:
        user = auth.create_user(
            email=credentials.email,
            password=credentials.password
        )
        return {"status": "success", "message": f"User {user.email} created successfully.", "uid": user.uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating user: {e}")

@app.post("/login")
async def login_user_api(credentials: UserCredentials):
    """
    Verifies a user exists in Firebase.
    NOTE: This is a simplified login for demo purposes. It does not verify the password.
    A production app would use Firebase Client SDKs to get an ID token and verify it here.
    """
    try:
        user = auth.get_user_by_email(credentials.email)
        # In a real app, you would verify the password or a token here.
        return {"status": "success", "message": "User verified.", "uid": user.uid, "email": user.email}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User not found or error logging in: {e}")

# =====================================================================
# REPLACE THE render_history_page FUNCTION IN app.py WITH THIS ONE
# =====================================================================

def render_history_page():
    st.title("📜 My Prediction History")
    
    # Call the new API endpoint to get history
    history_url = f"{API_URL}/history/{st.session_state['user_uid']}"
    
    try:
        response = requests.get(history_url, timeout=10)
        response.raise_for_status()
        
        results = response.json().get("history", [])
        
        if results:
            # Process the data for display
            display_data = []
            for item in results:
                details = item.get('house_details', {})
                display_data.append({
                    "Date": pd.to_datetime(item.get('timestamp')).strftime('%Y-%m-%d %H:%M'),
                    "City": details.get('City'),
                    "BHK": details.get('BHK'),
                    "Size": details.get('Size'),
                    "Predicted Rent (₹)": f"{item.get('prediction', 0):,.0f}"
                })
            
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        else:
            st.info("You have no prediction history yet. Go to the 'Rent Predictor' to make a prediction.")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch prediction history from the server. Error: {e}")
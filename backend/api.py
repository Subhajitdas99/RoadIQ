import sqlite3 # --- FIXED: Removed 'from dbm'
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from logic import process_frame
from database import DB_NAME, insert_log # --- FIXED: Removed duplicate DB_NAME
from geo_utils import get_location_details, get_municipal_authority
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Road Monitoring System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get-map-data")
def get_map_data():
    """Fetches all incident locations for the Streamlit Map"""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # NOTE: Ensure the table name 'reports' or 'road_logs' matches what insert_log uses in database.py
        # I have updated this to 'road_logs' which is the standard name we used previously. 
        # If your table is definitely named 'reports', change 'road_logs' back to 'reports' below.
        cursor.execute("SELECT id, timestamp, priority_level, damage_count, latitude, longitude, filename FROM road_logs ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "priority": row["priority_level"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "damage": row["damage_count"]
            })
        return results
    except Exception as e:
        # Return empty list instead of crashing if table doesn't exist yet
        print(f"Database error: {e}")
        return []

@app.post("/report-incident")
async def report_incident(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    # 1. SPAM CHECK
    if latitude == 0.0 or longitude == 0.0:
        raise HTTPException(status_code=400, detail="Spam Detected: Invalid GPS Coordinates")

    # 2. Location Validation
    in_india, address, city = get_location_details(latitude, longitude)
    
    if not in_india:
        raise HTTPException(status_code=400, detail=f"Location Error: {address}")

    # 3. Process Image
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Spam Detected: Empty Image File")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Spam Detected: Invalid Image Format")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    has_damage, severity, priority, save_path = process_frame(rgb_frame, file.filename, "API Upload")
    
    # 4. Determine Authority
    authority_name = "N/A"
    if has_damage:
        authority_name = get_municipal_authority(city)
    
    # 5. Save to Database
    insert_log("API Upload", file.filename, has_damage, severity, priority, save_path, 
               latitude, longitude, address, authority_name)
    
    return {
        "status": "Reported",
        "location": address,
        "priority": priority,
        "authority_notified": authority_name if has_damage else "None"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
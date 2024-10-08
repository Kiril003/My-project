from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import sqlite3
import json
import os

app = Flask(__name__)

DATABASE = 'user_permissions.db'
JSON_FILE = "C:\\stella-python\\user_permissions.json"

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('CREATE TABLE IF NOT EXISTS user_permissions (device_id TEXT PRIMARY KEY, expiration_date TEXT)')
        db.commit()
        sync_json_to_db()

def sync_json_to_db():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as file:
            data = json.load(file)
        db = get_db()
        for entry in data:
            db.execute('INSERT OR REPLACE INTO user_permissions (device_id, expiration_date) VALUES (?, ?)',
                       (entry['device_id'], entry['expiration_date']))
        db.commit()

def sync_db_to_json():
    db = get_db()
    rows = db.execute('SELECT device_id, expiration_date FROM user_permissions').fetchall()
    data = [{'device_id': row['device_id'], 'expiration_date': row['expiration_date']} for row in rows]
    with open(JSON_FILE, 'w') as file:
        json.dump(data, file, indent=4)

@app.route('/add_permission', methods=['POST'])
def add_permission():
    try:
        data = request.json
        device_id = data['device_id']
        duration = data['duration']
        
        duration_map = {
            '5m': timedelta(minutes=2),
            '1w': timedelta(weeks=1),
            '2w': timedelta(weeks=2),
            '1m': timedelta(days=30),
            '3m': timedelta(days=90),
            '6m': timedelta(days=180),
            '1y': timedelta(days=365)
        }
        expiration_date = (datetime.now() + duration_map[duration]).isoformat()
        
        db = get_db()
        db.execute('INSERT OR REPLACE INTO user_permissions (device_id, expiration_date) VALUES (?, ?)',
                   (device_id, expiration_date))
        db.commit()
        
        sync_db_to_json()
        
        return jsonify({"message": "Permission added successfully", "expiration_date": expiration_date}), 200
    except Exception as e:
        app.logger.error(f"Error adding permission: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/check_permission', methods=['POST'])
def check_permission():
    try:
        data = request.json
        device_id = data['device_id']
        
        db = get_db()
        result = db.execute('SELECT expiration_date FROM user_permissions WHERE device_id = ?', (device_id,)).fetchone()
        
        if result:
            expiration_date = datetime.fromisoformat(result['expiration_date'])
            if datetime.now() < expiration_date:
                return jsonify({"permission": True, "expiration_date": expiration_date.isoformat()}), 200
        
        return jsonify({"permission": False}), 200
    except Exception as e:
        app.logger.error(f"Error checking permission: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/list_permissions', methods=['GET'])
def list_permissions():
    try:
        db = get_db()
        result = db.execute('SELECT device_id, expiration_date FROM user_permissions').fetchall()
        
        permissions = [{"device_id": row['device_id'], "expiration_date": row['expiration_date']} for row in result]
        
        return jsonify(permissions), 200
    except Exception as e:
        app.logger.error(f"Error listing permissions: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/delete_permission', methods=['DELETE'])
def delete_permission():
    try:
        data = request.json
        device_id = data['device_id']
        
        db = get_db()
        now = datetime.now().isoformat()
        db.execute('INSERT OR REPLACE INTO user_permissions (device_id, expiration_date) VALUES (?, ?)',
                   (device_id, now))
        db.commit()
        
        sync_db_to_json()
        
        return jsonify({"deleted": True}), 200
    except Exception as e:
        app.logger.error(f"Error deleting permission: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    init_db()
    app.run(port=5000, debug=True)

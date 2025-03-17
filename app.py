from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
import numpy as np
from datetime import datetime
from functools import wraps
from bson import ObjectId

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# MongoDB configuration
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/forestfire')
mongo = PyMongo(app)

# Create indexes for email (unique)
with app.app_context():
    mongo.db.users.create_index('email', unique=True)
    # Create admin user if no users exist
    if not mongo.db.users.find_one():
        admin_user = {
            'name': 'Admin User',
            'email': 'rohitjadhav45074507@gmail.com',
            'password': generate_password_hash('zxcvbnm'),
            'is_admin': True,
            'created_at': datetime.utcnow()
        }
        mongo.db.users.insert_one(admin_user)
        print("Admin user created successfully!")

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or not user.get('is_admin'):
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('login.html')
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        email = data.get('email')
        password = data.get('password')
        is_admin = data.get('is_admin', False)
        
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password are required'})
        
        user = mongo.db.users.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):
            # For admin login, check if user is actually an admin
            if is_admin and not user.get('is_admin'):
                return jsonify({'success': False, 'error': 'This account does not have admin privileges'})
            
            session['user_id'] = str(user['_id'])
            session['user_name'] = user['name']
            session['is_admin'] = user.get('is_admin', False)
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Invalid email or password'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin')
@admin_required
def admin_dashboard():
    users = list(mongo.db.users.find())
    # Convert ObjectId to string for JSON serialization
    for user in users:
        user['_id'] = str(user['_id'])
    return render_template('admin.html', users=users)

@app.route('/admin/delete_user/<user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return jsonify({'success': False, 'error': 'User not found'})
    if user.get('is_admin'):
        return jsonify({'success': False, 'error': 'Cannot delete admin user'})
    try:
        mongo.db.users.delete_one({'_id': ObjectId(user_id)})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/toggle_admin/<user_id>', methods=['POST'])
@admin_required
def toggle_admin(user_id):
    try:
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'success': False, 'error': 'User not found'})
        
        new_admin_status = not user.get('is_admin', False)
        mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'is_admin': new_admin_status}}
        )
        return jsonify({'success': True, 'is_admin': new_admin_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not name or not email or not password:
            return jsonify({'success': False, 'error': 'All fields are required'})
        
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters long'})
        
        # Check if user already exists
        if mongo.db.users.find_one({'email': email}):
            return jsonify({'success': False, 'error': 'Email already registered'})
        
        # Create new user
        user = {
            'name': name,
            'email': email,
            'password': generate_password_hash(password),
            'is_admin': not mongo.db.users.find_one(),  # First user becomes admin
            'created_at': datetime.utcnow()
        }
        
        try:
            mongo.db.users.insert_one(user)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': 'Registration failed: Database error'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', user_name=session.get('user_name'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get latest model directory
        model_dirs = [d for d in os.listdir() if d.startswith('models_')]
        if not model_dirs:
            return jsonify({'success': False, 'error': 'No trained models found. Please train the models first.'})
        
        latest_model_dir = max(model_dirs)
        
        # Load the best model
        model = joblib.load(f'{latest_model_dir}/best_model.joblib')
        scaler = joblib.load(f'{latest_model_dir}/scaler.joblib')
        
        # Get data from form
        data = request.form.to_dict()
        
        # Convert month and day to numerical values
        months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        days = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
        
        # Prepare input data
        input_data = [
            float(data['X']), float(data['Y']),
            months[data['month']], days[data['day']],
            float(data['FFMC']), float(data['DMC']), float(data['DC']),
            float(data['ISI']), float(data['temp']), float(data['RH']),
            float(data['wind']), float(data['rain'])
        ]
        
        # Scale the input data
        input_scaled = scaler.transform([input_data])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Determine risk level
        if prediction < 1:
            risk_level = "Low Risk"
        elif prediction < 5:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'risk_level': risk_level
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run() 
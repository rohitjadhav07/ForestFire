from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
import numpy as np
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# For Vercel deployment, use PostgreSQL
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///users.db')
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

# Create database tables
with app.app_context():
    db.create_all()
    # Create admin user if no users exist
    if not User.query.first():
        admin_user = User(
            name='Admin User',
            email='rohitjadhav45074507@gmail.com',
            password=generate_password_hash('zxcvbnm'),
            is_admin=True
        )
        db.session.add(admin_user)
        db.session.commit()
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
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
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
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            # For admin login, check if user is actually an admin
            if is_admin and not user.is_admin:
                return jsonify({'success': False, 'error': 'This account does not have admin privileges'})
            
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['is_admin'] = user.is_admin
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Invalid email or password'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin')
@admin_required
def admin_dashboard():
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        return jsonify({'success': False, 'error': 'Cannot delete admin user'})
    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/toggle_admin/<int:user_id>', methods=['POST'])
@admin_required
def toggle_admin(user_id):
    user = User.query.get_or_404(user_id)
    try:
        user.is_admin = not user.is_admin
        db.session.commit()
        return jsonify({'success': True, 'is_admin': user.is_admin})
    except Exception as e:
        db.session.rollback()
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
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already registered'})
        
        # Create new user
        user = User(
            name=name,
            email=email,
            password=generate_password_hash(password),
            is_admin=not User.query.first()  # First user becomes admin
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
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
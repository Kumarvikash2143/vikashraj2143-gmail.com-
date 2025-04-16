from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import cv2
import numpy as np
from PIL import Image
import face_recognition
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pickle
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///face_recognition.db')
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    face_encodings = Column(LargeBinary)  # Stores serialized list of face encodings
    frame_count = Column(Integer, default=0)

# Create database tables
Base.metadata.create_all(engine)

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Please select an image file.'}), 400
        
        file = request.files['file']
        name = request.form.get('name')
        email = request.form.get('email')
        
        if not file or not name or not email:
            missing = []
            if not file: missing.append('photo')
            if not name: missing.append('name')
            if not email: missing.append('email')
            return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a JPG, JPEG, PNG, or GIF file.'}), 400
        
        try:
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save and process the image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Get face encoding
                image = face_recognition.load_image_file(filepath)
                face_locations = face_recognition.face_locations(image)
                
                if not face_locations:
                    return jsonify({'error': 'No face detected in the image. Please ensure your face is clearly visible and well-lit.'}), 400
                
                if len(face_locations) > 1:
                    return jsonify({'error': 'Multiple faces detected. Please upload an image with only your face.'}), 400
                
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                
                # Store in database
                session = Session()
                try:
                    # Check if user already exists
                    existing_user = session.query(User).filter_by(email=email).first()
                    if existing_user:
                        if existing_user.frame_count >= 100:
                            return jsonify({'error': 'Maximum number of frames (100) reached for this user.'}), 400
                        
                        # Add new encoding to existing ones
                        current_encodings = pickle.loads(existing_user.face_encodings)
                        
                        # Check for duplicate frames
                        for existing_encoding in current_encodings:
                            if face_recognition.compare_faces([existing_encoding], face_encoding)[0]:
                                return jsonify({'error': 'This frame appears to be too similar to an existing one. Please try a different pose or angle.'}), 400
                        
                        current_encodings.append(face_encoding)
                        existing_user.face_encodings = pickle.dumps(current_encodings)
                        existing_user.frame_count += 1
                        message = f'Frame {existing_user.frame_count} added successfully!'
                    else:
                        # Create new user
                        new_user = User(
                            name=name,
                            email=email,
                            face_encodings=pickle.dumps([face_encoding]),
                            frame_count=1
                        )
                        session.add(new_user)
                        message = 'First frame added successfully! Please add more frames from different angles for better recognition.'
                    
                    session.commit()
                    return jsonify({'success': True, 'message': message}), 200
                    
                except Exception as e:
                    session.rollback()
                    return jsonify({'error': f'Database error: {str(e)}'}), 500
                finally:
                    session.close()
                    
            except Exception as e:
                return jsonify({'error': f'Face recognition error: {str(e)}'}), 500
            
        except Exception as e:
            return jsonify({'error': f'Image processing error: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Clean up uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Please select an image file.'}), 400
        
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a JPG, JPEG, PNG, or GIF file.'}), 400
        
        try:
            # Save and process the image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Get face encoding from uploaded image
                unknown_image = face_recognition.load_image_file(filepath)
                face_locations = face_recognition.face_locations(unknown_image)
                
                if not face_locations:
                    return jsonify({'error': 'No face detected in the image. Please ensure your face is clearly visible and well-lit.'}), 400
                
                if len(face_locations) > 1:
                    return jsonify({'error': 'Multiple faces detected. Please upload an image with only your face.'}), 400
                
                unknown_encoding = face_recognition.face_encodings(unknown_image, face_locations)[0]
                
                # Compare with stored encodings
                session = Session()
                try:
                    best_match = None
                    best_match_score = 0
                    
                    for user in session.query(User).all():
                        stored_encodings = pickle.loads(user.face_encodings)
                        matches = face_recognition.compare_faces(stored_encodings, unknown_encoding)
                        if True in matches:
                            match_score = sum(matches) / len(matches)
                            if match_score > best_match_score:
                                best_match_score = match_score
                                best_match = user
                    
                    if best_match and best_match_score >= 0.5:
                        return jsonify({
                            'found': True,
                            'name': best_match.name,
                            'email': best_match.email,
                            'confidence': float(best_match_score)
                        })
                    
                    return jsonify({'found': False, 'error': 'No matching user found in the database.'})
                    
                except Exception as e:
                    return jsonify({'error': f'Database error: {str(e)}'}), 500
                finally:
                    session.close()
                    
            except Exception as e:
                return jsonify({'error': f'Face recognition error: {str(e)}'}), 500
            
        except Exception as e:
            return jsonify({'error': f'Image processing error: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Clean up uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

@app.route('/database')
def view_database():
    session = Session()
    try:
        users = session.query(User).all()
        user_data = []
        
        for user in users:
            face_encodings = pickle.loads(user.face_encodings)
            
            user_info = {
                'name': user.name,
                'email': user.email,
                'frame_count': user.frame_count,
                'images': []
            }
            
            # Convert face encodings to more visually appealing images
            for i, encoding in enumerate(face_encodings):
                # Create a larger image with white background
                img = np.full((500, 500, 3), 245, dtype=np.uint8)  # Lighter background
                
                # Create a visualization grid
                grid_size = int(np.sqrt(len(encoding)))
                cell_size = 500 // grid_size
                
                for j, value in enumerate(encoding):
                    if j >= grid_size * grid_size:
                        break
                    
                    # Enhanced normalization for better contrast
                    color = int(max(40, min(220, ((value + 1) / 2) * 255)))
                    
                    # Calculate grid position
                    row = j // grid_size
                    col = j % grid_size
                    
                    # Draw a filled circle with better visibility
                    center = (col * cell_size + cell_size//2, 
                            row * cell_size + cell_size//2)
                    radius = int(cell_size * 0.35)  # Slightly smaller circles for better spacing
                    
                    # Draw a filled circle with gradient effect
                    cv2.circle(img, center, radius, (color, color, color), -1)
                    # Add a darker border for better definition
                    border_color = max(0, color - 60)
                    cv2.circle(img, center, radius, (border_color, border_color, border_color), 2)
                
                # Add a subtle grid background
                for x in range(grid_size + 1):
                    cv2.line(img, (x * cell_size, 0), (x * cell_size, 500), (230, 230, 230), 1)
                    cv2.line(img, (0, x * cell_size), (500, x * cell_size), (230, 230, 230), 1)
                
                # Add a border to the entire image
                cv2.rectangle(img, (0, 0), (499, 499), (180, 180, 180), 2)
                
                # Add frame number with better styling
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f'Frame {i + 1}'
                # Add background for text
                text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
                cv2.rectangle(img, (8, 8), (text_size[0] + 12, 35), (255, 255, 255), -1)
                cv2.rectangle(img, (8, 8), (text_size[0] + 12, 35), (180, 180, 180), 1)
                cv2.putText(img, text, (10, 30), font, 0.8, (60, 60, 60), 2)
                
                # Convert to base64 with high quality
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                user_info['images'].append({
                    'index': i + 1,
                    'data': img_base64
                })
            
            user_data.append(user_info)
            
        return render_template('database.html', users=user_data)
    except Exception as e:
        flash(f'Error loading database: {str(e)}')
        return redirect(url_for('index'))
    finally:
        session.close()

if __name__ == '__main__':
    app.run(debug=True) 
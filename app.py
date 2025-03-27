import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
import threading
import time
from pathlib import Path
from utils.video_standardizer import standardize_video, extract_fixed_frames
from dataloader import get_transforms
from PIL import Image
from torchvision import transforms
from main import initialize_model
from evaluate_ensemble import MODEL_CONFIGS, setup_device
import torch.nn.functional as F

app = Flask(__name__)
app.secret_key = "violence_detection_app"
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
current_video = None
current_results = None
processing_lock = threading.Lock()
device = setup_device(0)  # Use GPU 0
models = {}
transform = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_models():
    """Load the ensemble models"""
    global models, device
    
    model_types = ['transformer', '3d_cnn', '2d_cnn_lstm']
    model_weights = {
        'transformer': 2.5,
        '3d_cnn': 2.4,
        '2d_cnn_lstm': 2.3
    }
    
    print("Loading models...")
    for model_type in model_types:
        try:
            # Initialize model with correct configuration
            model_params = MODEL_CONFIGS.get(model_type, {'num_classes': 2})
            model = initialize_model(model_type, device, **model_params)
            
            # Load weights
            model_dir = f"./output/{model_type}"
            checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
            best_checkpoint = next((f for f in checkpoint_files if 'best' in f), None)
            
            if best_checkpoint:
                checkpoint_path = os.path.join(model_dir, best_checkpoint)
                print(f"Loading {model_type} from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                    
                model.eval()
                models[model_type] = {"model": model, "weight": model_weights[model_type]}
                print(f"Successfully loaded {model_type}")
            else:
                print(f"No checkpoint found for {model_type}")
        except Exception as e:
            print(f"Error loading {model_type}: {e}")
    
    print(f"Loaded {len(models)} models")

def process_video(video_path):
    """Process the uploaded video and get predictions"""
    global models, device, transform
    
    try:
        # Standardize the video to 224x224, 15fps
        print("Standardizing video...")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_standardized.mp4")
        stats = standardize_video(video_path, temp_path, 
                                target_width=224, target_height=224, 
                                fps=15)
        if not stats:
            return {"error": "Failed to standardize video"}
        
        # Extract frames for prediction
        print("Extracting frames...")
        frames = extract_fixed_frames(temp_path, num_frames=16, resize_dim=(224, 224))
        if frames is None or len(frames) < 16:
            return {"error": "Failed to extract enough frames"}
        
        # Process frames
        processed_frames = []
        for frame in frames:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if transform:
                pil_image = transform(pil_image)
            processed_frames.append(pil_image)
        
        # Create batch
        frames_tensor = torch.stack(processed_frames)
        # For 3D CNN format
        frames_3d = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        # For 2D CNN+LSTM format
        frames_2d = frames_tensor.unsqueeze(0)  # [1, T, C, H, W]
        
        # Run inference with each model
        print("Running inference...")
        all_probs = []
        all_weights = []
        
        for model_name, model_info in models.items():
            model = model_info["model"]
            weight = model_info["weight"]
            
            with torch.no_grad():
                if model_name == '3d_cnn':
                    outputs = model(frames_3d.to(device))
                else:  # transformer and 2d_cnn_lstm use [B, T, C, H, W]
                    outputs = model(frames_2d.to(device))
                
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_weights.append(weight)
        
        # Compute weighted ensemble
        ensemble_probs = np.zeros_like(all_probs[0])
        total_weight = sum(all_weights)
        
        for i, (probs, weight) in enumerate(zip(all_probs, all_weights)):
            ensemble_probs += probs * (weight / total_weight)
        
        # Extract probabilities for visualization
        class_probs = {
            "peace": float(ensemble_probs[0][0]),
            "violence": float(ensemble_probs[0][1])
        }
        predicted_class = "Violence" if class_probs["violence"] > class_probs["peace"] else "No Violence"
        
        # Get individual model predictions
        model_predictions = {}
        for i, (model_name, model_info) in enumerate(models.items()):
            model_predictions[model_name] = {
                "peace": float(all_probs[i][0][0]),
                "violence": float(all_probs[i][0][1])
            }
        
        return {
            "success": True,
            "class_probs": class_probs,
            "predicted_class": predicted_class,
            "model_predictions": model_predictions
        }
    
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error processing video: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_video, current_results, processing_lock
    
    # Check if a file was uploaded
    if 'video' not in request.files:
        flash('No video file provided')
        return redirect(request.url)
    
    file = request.files['video']
    
    # If the user doesn't select a file
    if file.filename == '':
        flash('No video selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        with processing_lock:
            # Save the video
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the video
            results = process_video(filepath)
            
            if "error" in results:
                flash(results["error"])
                return redirect(url_for('index'))
            
            # Update globals
            current_video = filepath
            current_results = results
            
            return redirect(url_for('results'))
    else:
        flash('Invalid file format. Please upload mp4, avi, mov, or mkv.')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    global current_video, current_results
    
    if current_video is None or current_results is None:
        flash('No video has been processed yet')
        return redirect(url_for('index'))
    
    # Convert video to base64 for embedding or use a video serving endpoint
    video_filename = os.path.basename(current_video)
    
    return render_template(
        'results.html', 
        video_filename=video_filename,
        results=current_results
    )

@app.route('/video/<filename>')
def serve_video(filename):
    """Stream the video file"""
    def generate():
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield b''
            return
        
        while True:
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Add confidence scores overlay
            if current_results:
                peace_prob = current_results["class_probs"]["peace"] * 100
                violence_prob = current_results["class_probs"]["violence"] * 100
                
                # Draw background for text
                cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)
                
                # Add text
                cv2.putText(frame, f"Prediction: {current_results['predicted_class']}", 
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Peace: {peace_prob:.1f}%", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Violence: {violence_prob:.1f}%", 
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1/30)  # Control frame rate
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize transforms
    _, transform = get_transforms()
    # Load models
    load_models()
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
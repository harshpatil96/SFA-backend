from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
import time
import os
import json
from datetime import datetime
import cv2
import traceback

app = Flask(__name__)
CORS(app)

# Load configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'grape_disease_model.h5')
DISEASE_DATA_PATH = os.path.join(BASE_DIR, 'models', 'disease_data.json')

print(f"\n{'='*60}")
print("🌿 SMART FARMING - MULTI-CROP DISEASE PREDICTION API")
print(f"{'='*60}")

# Load model
try:
    # Keras 3 to Keras 2 Compatibility Patch
    import h5py
    import json
    with h5py.File(MODEL_PATH, 'a') as f:
        if 'model_config' in f.attrs:
            val = f.attrs['model_config']
            config_str = val.decode('utf-8') if isinstance(val, bytes) else str(val)
            config = json.loads(config_str)
            has_changes = False
            for layer in config.get('config', {}).get('layers', []):
                if 'config' in layer:
                    if 'batch_shape' in layer['config']:
                        layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                        has_changes = True
                    if 'optional' in layer['config']:
                        layer['config'].pop('optional')
                        has_changes = True
            if has_changes:
                print("✓ Patched Keras 3 InputLayer for Keras 2 compatibility")
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
    print(f"✓ Model Output Shape: {model.output_shape}")
except Exception as e:
    import sys
    try:
        file_size = os.path.getsize(MODEL_PATH)
        MODEL_ERROR = f"Exception: {str(e)} | File size: {file_size} bytes."
        if file_size < 1000000:
            with open(MODEL_PATH, 'rb') as f:
                head = f.read(100)
                MODEL_ERROR += f" File header: {head}"
    except Exception as fe:
        MODEL_ERROR = f"Exception: {str(e)} | Could not read file info: {fe}"
    print(f"✗ Error loading model: {MODEL_ERROR}")
    model = None

# Verified class list matching the grape model and dataset directory order
CLASS_NAMES = [
    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy'
]

# Load disease data dynamically
DISEASE_DATA = {}
try:
    if os.path.exists(DISEASE_DATA_PATH):
        try:
            with open(DISEASE_DATA_PATH, 'r', encoding='utf-8') as f:
                DISEASE_DATA = json.load(f)
        except Exception:
            # Fallback to UTF-16 in case Windows generated the file
            with open(DISEASE_DATA_PATH, 'r', encoding='utf-16') as f:
                DISEASE_DATA = json.load(f)
        print(f"✓ Disease data loaded: {len(DISEASE_DATA)} entries")
    else:
        print(f"⚠ Disease data file not found at {DISEASE_DATA_PATH}")
except Exception as e:
    print(f"✗ Error loading disease data: {e}")

print(f"✓ API Server ready at: http://localhost:5000")
print(f"{'='*60}\n")

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'Grape Disease Detection API',
        'version': '2.0',
        'status': 'Active',
        'model': 'EfficientNetB0',
        'accuracy': '94.8%'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'Healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'model_error': MODEL_ERROR if 'MODEL_ERROR' in globals() else None,
        'disease_data_loaded': len(DISEASE_DATA) > 0
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        start_time = time.time()
        
        # Validate model
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Robust Image Decoding (Multi-stage)
        try:
            import io
            from PIL import Image
            
            # 1. Read the bytes once and keep them
            file.seek(0)
            img_bytes = file.read()
            
            print(f"\n[DEBUG] --- Image Diagnosis ---", flush=True)
            print(f"[DEBUG] Filename: {file.filename}", flush=True)
            print(f"[DEBUG] Byte count: {len(img_bytes)} bytes", flush=True)
            
            if len(img_bytes) < 10:
                return jsonify({'error': f'File too small or empty ({len(img_bytes)} bytes)'}), 400
            
            header_hex = img_bytes[:16].hex(' ')
            print(f"[DEBUG] Header (Hex): {header_hex}", flush=True)
            
            img = None
            decode_errors = []
            
            # Stage 1: Try PIL
            try:
                img = Image.open(io.BytesIO(img_bytes))
                img.verify() # Verify it's not truncated
                img = Image.open(io.BytesIO(img_bytes)) # Re-open for processing
                print("[DEBUG] Stage 1: PIL successfully identified image", flush=True)
            except Exception as e:
                decode_errors.append(f"PIL: {str(e)}")
                print(f"[DEBUG] Stage 1: PIL failed - {e}", flush=True)
            
            # Stage 2: Try OpenCV Fallback
            if img is None:
                try:
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if cv_img is not None:
                        # Convert BGR to RGB
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(cv_img)
                        print("[DEBUG] Stage 2: OpenCV successfully decoded image", flush=True)
                    else:
                        decode_errors.append("OpenCV: imdecode returned None")
                except Exception as e:
                    decode_errors.append(f"OpenCV Error: {str(e)}")
                    print(f"[DEBUG] Stage 2: OpenCV failed - {e}", flush=True)
            
            if img is None:
                error_details = " | ".join(decode_errors)
                
                # CRITICAL: Save failing image for diagnosis
                try:
                    debug_file = os.path.join(BASE_DIR, 'last_received_error.bin')
                    with open(debug_file, 'wb') as f:
                        f.write(img_bytes)
                    print(f"[DEBUG] Saved failing image to: {debug_file}", flush=True)
                except:
                    pass

                return jsonify({
                    'error': f'Image decoding failed. {error_details}',
                    'debug': {
                        'size': len(img_bytes),
                        'header': header_hex,
                        'first_100_bytes': img_bytes[:100].hex(' ')
                    }
                }), 400

            # Final processing
            img = img.resize((224, 224))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = img_to_array(img)
            # EfficientNetB0 does internal rescaling
            # img_array = img_array / 255.0  # Removed division
            img_array = np.expand_dims(img_array, axis=0)
        except Exception as e:
            return jsonify({'error': f'Image processing failed: {str(e)}'}), 400
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Safety check: Match predictions to CLASS_NAMES
        num_model_classes = model.output_shape[-1]
        num_class_names = len(CLASS_NAMES)
        
        if num_model_classes != num_class_names:
            print(f"⚠ Model expects {num_model_classes} classes but found {num_class_names}. Syncing...", flush=True)
            active_names = CLASS_NAMES[:num_model_classes]
        else:
            active_names = CLASS_NAMES

        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = active_names[predicted_class_idx] if predicted_class_idx < len(active_names) else "Unknown"

        
        # Get disease info and parse crop/disease
        raw_class = predicted_class
        crop_name = "Unknown"
        disease_display_name = raw_class
        
        if "___" in raw_class:
            parts = raw_class.split("___")
            crop_name = parts[0].replace("_", " ")
            disease_display_name = parts[1].replace("_", " ")

        disease_info = DISEASE_DATA.get(raw_class, {
            'name': disease_display_name,
            'severity': 'Unknown',
            'urgency': 'Consult specialist',
            'causes': 'Analysis pending for this crop/disease combination.',
            'symptoms': 'Specific symptoms for this variety are being cataloged.',
            'treatment': 'Standard agricultural best practices recommended.',
            'fertilizer': 'Basic NPK balance recommended.'
        })
        
        # Calculate analysis time
        analysis_time = int((time.time() - start_time) * 1000)
        
        # Prepare all predictions
        all_predictions = []
        for idx, class_name in enumerate(active_names):
            conf = float(predictions[0][idx])
            
            # Get display name for all predictions
            if class_name in DISEASE_DATA:
                display_name = DISEASE_DATA[class_name]['name']
            else:
                display_name = class_name.split("___")[-1].replace("_", " ") if "___" in class_name else class_name
                
            all_predictions.append({
                'disease': display_name,
                'confidence': float(round(conf * 100, 1)),
                'class': class_name
            })

        
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'crop': crop_name,
            'disease': {
                'name': disease_info['name'],
                'class': raw_class,
                'confidence': float(round(confidence * 100, 1)),
                'severity': disease_info.get('severity', 'Unknown'),
                'urgency': disease_info.get('urgency', 'Contact expert')
            },
            'analysis': {
                'causes': disease_info.get('causes', 'Unknown'),
                'symptoms': disease_info.get('symptoms', 'Unknown'),
                'treatment': disease_info.get('treatment', 'Consult agricultural expert'),
                'fertilizer': disease_info.get('fertilizer', 'Consult expert')
            },
            'all_predictions': all_predictions,

            'model': {
                'name': 'EfficientNetB0',
                'accuracy': '94.8%',
                'analysis_time_ms': analysis_time
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\n[ERROR] Prediction failed:\n{error_msg}", flush=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/disease-info', methods=['GET'])
def get_disease_info():
    """Get all disease information"""
    try:
        response = {}
        for class_name in CLASS_NAMES:
            disease_info = DISEASE_DATA.get(class_name, {})
            response[class_name] = disease_info
        
        return jsonify({
            'success': True,
            'diseases': response,
            'total_diseases': len(CLASS_NAMES)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'success': True,
        'model': {
            'name': 'EfficientNetB0',
            'framework': 'TensorFlow 2.x',
            'training_accuracy': '95.2%',
            'validation_accuracy': '94.8%',
            'image_size': '224x224',
            'input_format': 'RGB Image',
            'classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES
        },
        'training': {
            'phase1_epochs': 20,
            'phase2_epochs': 30,
            'batch_size': 32,
            'augmentation': 'Strong (rotation, zoom, flip, brightness)',
            'optimizer': 'Adam',
            'learning_rate_phase1': 0.001,
            'learning_rate_phase2': 0.00001
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

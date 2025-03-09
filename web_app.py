import os
import json
import asyncio
import threading
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

# Import your existing functionality
from app import (
    process_images, 
    load_state, 
    log as app_log, 
    current_start_number, 
    is_scanning,
    output_dir
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        RotatingFileHandler('web_app.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Background task handling
background_thread = None
stop_background_thread = False

def run_background_scanner():
    """Background thread to periodically scan for new comics"""
    global stop_background_thread
    
    logger.info("Starting background scanner thread")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not stop_background_thread:
            # Run the scan process
            loop.run_until_complete(process_images())
            
            # Sleep for 5 minutes before next scan
            for _ in range(300):  # 5 minutes in seconds
                if stop_background_thread:
                    break
                time.sleep(1)
                
    except Exception as e:
        logger.error(f"Background scanner error: {str(e)}")
    finally:
        logger.info("Background scanner thread stopped")
        loop.close()

def start_background_scanner():
    """Start the background scanner if not already running"""
    global background_thread, stop_background_thread
    
    if background_thread is None or not background_thread.is_alive():
        stop_background_thread = False
        background_thread = threading.Thread(target=run_background_scanner)
        background_thread.daemon = True
        background_thread.start()
        logger.info("Background scanner started")
        return True
    else:
        logger.info("Background scanner already running")
        return False

def stop_background_scanner():
    """Stop the background scanner"""
    global background_thread, stop_background_thread
    
    if background_thread and background_thread.is_alive():
        stop_background_thread = True
        logger.info("Stopping background scanner (may take a moment)")
        return True
    else:
        logger.info("No background scanner to stop")
        return False

# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Get current scanner status"""
    state = load_state()
    
    return jsonify({
        'isScanning': is_scanning,
        'currentStartNumber': current_start_number,
        'backgroundScannerActive': background_thread is not None and background_thread.is_alive(),
        'lastScanInfo': state.get('lastStartNumber', current_start_number),
        'downloadedCount': len(state.get('downloadedChapters', {}))
    })

@app.route('/api/downloads')
def get_downloads():
    """Get list of downloaded comics"""
    state = load_state()
    downloaded = state.get('downloadedChapters', {})
    
    # Format for display
    formatted_downloads = []
    for key, data in downloaded.items():
        try:
            series_name, chapter = key.rsplit('_', 1)
            formatted_downloads.append({
                'seriesName': series_name,
                'chapter': chapter,
                'downloadedAt': data.get('timestamp', 'Unknown'),
                'imageCount': data.get('imageCount', 0),
                'path': data.get('path', '').replace(output_dir, '').lstrip('/\\'),
            })
        except Exception as e:
            logger.error(f"Error formatting download entry {key}: {str(e)}")
    
    # Sort by series name and then by chapter
    formatted_downloads.sort(key=lambda x: (x['seriesName'], x['chapter']))
    
    return jsonify(formatted_downloads)

@app.route('/api/scan/start')
def start_scan():
    """Start background scanner"""
    success = start_background_scanner()
    return jsonify({'success': success, 'message': 'Background scanner started' if success else 'Scanner already running'})

@app.route('/api/scan/stop')
def stop_scan():
    """Stop background scanner"""
    success = stop_background_scanner()
    return jsonify({'success': success, 'message': 'Background scanner stopping' if success else 'No scanner running'})

@app.route('/api/scan/once', methods=['POST'])
def scan_once():
    """Run one scan immediately"""
    start_number = request.json.get('startNumber', current_start_number)
    
    try:
        # Run in a separate thread to not block the web server
        def run_scan():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_images())
            loop.close()
        
        scan_thread = threading.Thread(target=run_scan)
        scan_thread.daemon = True
        scan_thread.start()
        
        return jsonify({'success': True, 'message': f'Manual scan started from {start_number}'})
    except Exception as e:
        logger.error(f"Manual scan error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/comics/<path:filepath>')
def serve_comic(filepath):
    """Serve comic files"""
    return send_from_directory(output_dir, filepath)

# Start the app
if __name__ == '__main__':
    # Start background scanner on startup
    start_background_scanner()
    # Run Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)

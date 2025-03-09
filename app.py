import os
import json
import time
import requests
import easyocr
import numpy as np
import re
from datetime import datetime
from PIL import Image
from io import BytesIO
import asyncio
import aiohttp
import shutil

# Configuration
proxy_url = 'https://scythe-elfin-basilisk.glitch.me/proxy?url='
base_urls = [
    "https://gg.asuracomic.net/storage/media/*/conversions/#-optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/#-kopya-optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/#-result_optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/#.jpg"
]

zero_page_base_urls = [
    "https://gg.asuracomic.net/storage/media/*/conversions/00-optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/00-kopya-optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/00-result_optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/00.jpg",
    "https://gg.asuracomic.net/storage/media/*/conversions/0000-copy-optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/00000-copy-optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/000-optimized.webp",
    "https://gg.asuracomic.net/storage/media/*/conversions/00-copy-optimized.webp"
]

# Known series list
known_series_list = [
    "Reaper of the Drifting Moon",
    "Leveling with the Gods",
    "Solo Max-Level Newbie",
    "The Great Mage Returns After 4000 Years",
    "Nano Machine",
    "Legend of the Northern Blade",
    "Return of the Mount Hua Sect",
    "Heavenly Demon Cultivation Simulation",
    "The Beginning After the End",
    "Martial Peak",
    "Overgeared",
    "Solo Leveling",
    "Otherworldly Sword King's Survival Records",
    "Second Life Ranker",
    "Omniscient Reader's Viewpoint",
    "The Return of the Crazy Demon",
    "Chronicles of the Heavenly Demon",
    "Return of the Frozen Player",
    "Tower of God",
    "God of Blackfield",
    "Regressor Instruction Manual",
    "Weapon Maker",
    "The Nebula's Civilization",
    "Dragon-Devouring Mage",
    "Regressing with the King's Power",
    "Genius Martial Arts Trainer",
    "The Heavenly Demon Wants a Quiet Life",
    "Emperor of Steel",
    "A Dragonslayer's Peerless Regression",
    "Reincarnator",
    "Youngest Scion of the Mages",
    "Myst, Might, Mayhem",
    "The Ultimate Shut-in",
    "Standard of Reincarnation",
    "I Am the Fated Villain",
    "The Terminally Ill Young Master",
    "The World After The End",
    "The Priest of Corruption",
    "Overpowered Sword",
    "Infinite Mage",
    "Murim Login",
    "SSS-Class Suicide Hunter",
    "Villain to Kill",
    "Doctor's Rebirth",
    "Your Talent is Mine",
    "The Indomitable Martial King",
    "Absolute Regression",
    "The Regressed Mercenary's Machine",
    "Night of Soulless Heathens",
    "Reborn as the Enemy Prince",
    "Artifact-Devouring Player",
    "Love Letter From The Future",
    "Genius Archer's Streaming",
    "The Greatest Estate Developer",
    "The Tutorial is Too Hard",
    "The Regressed Son of a Duke Is an Assassin",
    "Revenge of the Iron-Blooded Swordhound",
    "I'm Gonna Annihilate This Land",
    "Childhood Friend of the Zenith",
    "Logging 10,000 Years into the Future",
    "Light of Arad: Forerunner",
    "Somebody Stop the Pope",
    "The Knight King Who Returned with a God",
    "Boundless Necromancer",
    "Academy's Genius Swordmaster",
    "The Last Adventurer",
    "The Hero Returns",
    "Legend of the Reincarnated Demon",
    "The Extra's Academy Survival Guide",
    "Margrave's Bastard Son was The Emperor"
]
# Global variables
current_start_number = 279290  # Starting search number
is_scanning = False
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state.json')
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloader.log')

# Create download directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize EasyOCR reader
reader = None

def initialize_ocr():
    global reader
    if reader is None:
        log("Initializing EasyOCR reader...", 'info')
        reader = easyocr.Reader(['en'])  # Initialize for English
        log("EasyOCR initialized", 'success')
    return reader

# Load previous state if exists
def load_state():
    global current_start_number
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                current_start_number = state.get('lastStartNumber', current_start_number)
                log(f"Loaded previous state: starting from number {current_start_number}")
                return state
    except Exception as error:
        log(f"Error loading state: {str(error)}", 'error')
    return {"lastStartNumber": current_start_number, "downloadedChapters": {}}

# Save state
def save_state(state):
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        log(f"State saved: last start number {state['lastStartNumber']}")
    except Exception as error:
        log(f"Error saving state: {str(error)}", 'error')

# Logging function
def log(message, type='info'):
    timestamp = datetime.now().isoformat()
    log_message = f"[{timestamp}] [{type.upper()}] {message}"
    print(log_message)
    
    # Append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')

# Check URL validity
async def check_url(url, session):
    try:
        full_url = url if url.startswith('http') else f"{proxy_url}{url}"
        async with session.head(full_url) as response:
            return response.status == 200
    except Exception as error:
        log(f"Error checking URL {url}: {str(error)}", 'error')
        return False

# Extract text from image using EasyOCR with ROI
async def extract_text_from_image(image_url, session):
    log(f"Starting OCR text extraction from image: {image_url} using EasyOCR")
    
    # Create temp directory for OCR files
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_ocr')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_image_path = os.path.join(temp_dir, 'temp_ocr_image.webp')
    converted_image_path = os.path.join(temp_dir, 'temp_ocr_image.jpg')
    
    try:
        # Download the image
        full_url = image_url if image_url.startswith('http') else f"{proxy_url}{image_url}"
        log(f"Fetching image from: {full_url}")
        
        async with session.get(full_url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch image: {response.status} {response.reason}")
            
            image_data = await response.read()
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
        
        log('Image downloaded, preparing for OCR processing...')

        # Convert to JPEG format (more reliable for OCR)
        with Image.open(temp_image_path) as img:
            img = img.convert('RGB')
            img.save(converted_image_path, 'JPEG')

        log('Image converted to JPEG, running OCR with EasyOCR...')
        
        # Initialize EasyOCR if not already initialized
        ocr_reader = initialize_ocr()
        
        # Get image dimensions for calculating ROI
        with Image.open(converted_image_path) as img:
            image_width, image_height = img.size
        
        # Calculate ROI based on typical manga/comic title positions
        # Focus on the top portion of the image where titles are usually found
        roi_x = 531              
        roi_y = 170                                     
        roi_width = 551            
        roi_height = 279        
        
        log(f"Using ROI: x={roi_x}, y={roi_y}, width={roi_width}, height={roi_height}")
        
        # Crop the image to ROI for better OCR performance
        roi_image_path = os.path.join(temp_dir, 'roi_ocr_image.jpg')
        with Image.open(converted_image_path) as img:
            roi_img = img.crop((roi_x, roi_y, roi_x + roi_width, roi_y + roi_height))
            roi_img.save(roi_image_path)
        
        # Run EasyOCR on the ROI
        log('Starting EasyOCR processing on ROI...')
        results = ocr_reader.readtext(roi_image_path)
        
        # If ROI gives no results, fall back to full image
        ocr_results = results
        if not results or len(results) == 0:
            log('No text found in ROI, falling back to full image', 'warning')
            ocr_results = ocr_reader.readtext(converted_image_path)
        
        # Parse results
        extracted_text = ''
        confidence = 0
        results_count = 0
        
        if ocr_results and len(ocr_results) > 0:
            for result in ocr_results:
                if result and len(result) >= 2:
                    bbox, text, conf = result if len(result) == 3 else (result[0], result[1], 0.5)
                    extracted_text += ' ' + text
                    confidence += conf
                    results_count += 1
            
            if results_count > 0:
                confidence = confidence / results_count
        
        extracted_text = extracted_text.strip()
        
        log(f"OCR completed with average confidence: {(confidence * 100):.2f}%")
        log(f"Extracted text: \"{extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}\"")
        
        # Extract series name from the text
        series_name = find_series_name_in_text(extracted_text)
        
        # Extract chapter number using regex
        chapter_regex = re.compile(r'chapter\s*(\d+)', re.IGNORECASE)
        ch_match = chapter_regex.search(extracted_text)
        chapter_number = ch_match.group(1) if ch_match else "unknown"
        
        # If we couldn't find the chapter number, look for it in standard formats
        if chapter_number == "unknown":
            alt_formats = [
                re.compile(r'ch\.?\s*(\d+)', re.IGNORECASE),     # Ch. 123
                re.compile(r'episode\s*(\d+)', re.IGNORECASE),   # Episode 123
                re.compile(r'ep\.?\s*(\d+)', re.IGNORECASE),      # Ep. 123
                re.compile(r'#\s*(\d+)', re.IGNORECASE),          # #123
                re.compile(r'\[(\d+)\]'),                        # [123]
                re.compile(r'(\d+)\s*:')                         # 123:
            ]
            
            for regex in alt_formats:
                match = regex.search(extracted_text)
                if match:
                    chapter_number = match.group(1)
                    break
        
        # Try to extract from the first number in the text if all else fails
        if chapter_number == "unknown":
            number_match = re.search(r'\d+', extracted_text)
            if number_match:
                chapter_number = number_match.group(0)
                log('Used fallback number extraction method', 'info')
        
        # Use default values if detection failed
        if not series_name:
            # Try to get series name by filename pattern analysis
            url_parts = image_url.split('/')
            filename_part = url_parts[-2] if len(url_parts) > 2 else ''
            
            # Look for patterns in the URL
            for series in known_series_list:
                simplified_series = series.lower().replace(' ', '')
                simplified_filename = filename_part.lower()
                
                if simplified_series in simplified_filename:
                    series_name = series
                    log(f'Detected series "{series_name}" from URL pattern', 'success')
                    break
            
            if not series_name:
                series_name = "Unknown Series"
                log('Could not detect series name from OCR or URL, using default value', 'error')
        
        log(f'Series detected: "{series_name}", Chapter: {chapter_number}', 'success')
        
        return {
            "seriesName": series_name,
            "chapterNumber": chapter_number
        }
    except Exception as error:
        log(f"OCR extraction error: {str(error)}", 'error')
        # Fallback to default values if OCR fails
        return {
            "seriesName": "Unknown Series",
            "chapterNumber": "unknown"
        }
    finally:
        # Clean up temp files
        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if os.path.exists(converted_image_path):
                os.remove(converted_image_path)
            if os.path.exists(os.path.join(temp_dir, 'roi_ocr_image.jpg')):
                os.remove(os.path.join(temp_dir, 'roi_ocr_image.jpg'))
            
            # Only attempt to remove directory if it exists and is empty
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_err:
            log(f"Warning: Failed to clean up some temp files: {str(cleanup_err)}", 'warning')

# Find the best matching series name in the extracted text
def find_series_name_in_text(text):
    if not text:
        return None
    
    # Convert the text to lowercase for case-insensitive matching
    lowercase_text = text.lower()
    
    # First try exact matches with known series
    for series in known_series_list:
        if series.lower() in lowercase_text:
            return series
    
    # If no exact match, try fuzzy matching
    best_match = None
    highest_match_score = 0
    
    for series in known_series_list:
        score = calculate_match_score(lowercase_text, series.lower())
        if score > highest_match_score and score > 0.6:  # Threshold of 60% match
            highest_match_score = score
            best_match = series
    
    return best_match or None

# Calculate a simple match score between text and a search term
def calculate_match_score(text, search_term):
    # Remove spaces for more lenient matching
    simplified_text = text.replace(' ', '')
    simplified_term = search_term.replace(' ', '')
    
    # If the term is found completely, high score
    if simplified_term in simplified_text:
        return 1.0
    
    # Count how many words match
    words = search_term.split()
    matched_words = 0
    
    for word in words:
        if len(word) > 2 and word in text:  # Only count words longer than 2 chars
            matched_words += 1
    
    # Calculate word match percentage
    if len(words) > 0:
        return matched_words / len(words)
    
    return 0

# Find valid images batch
async def find_valid_start_and_frames(start_number, session):
    log(f"Starting search from number: {start_number}")
    number = start_number
    valid_start_number = None
    valid_frames = 0
    valid_urls = []
    zero_page_exists = False
    zero_page_urls = []
    max_attempts = 100
    attempts = 0

    # Step 1: Find the first valid start number for frame 01
    while attempts < max_attempts:
        attempts += 1
        log(f"Checking start number {number} (attempt {attempts}/{max_attempts})")

        # Use the `01` frame for checking the start number
        urls_to_check = [url.replace('*', str(number)).replace('#', '01') for url in base_urls]

        results = await asyncio.gather(*[check_url(url, session) for url in urls_to_check])

        if any(results):
            valid_start_number = number
            log(f"Found valid start number: {valid_start_number}", 'success')

            # Check for 00 page
            zero_number = valid_start_number - 1
            zero_urls_to_check = [url.replace('*', str(zero_number)) for url in zero_page_base_urls]

            log(f"Checking for zero page at number: {zero_number}")
            zero_results = await asyncio.gather(*[check_url(url, session) for url in zero_urls_to_check])

            zero_page_exists = any(zero_results)
            zero_page_urls = [url for url, is_valid in zip(zero_urls_to_check, zero_results) if is_valid]

            if zero_page_exists:
                log(f"Found zero page at number: {zero_number}", 'success')
            else:
                log(f"No zero page found at number: {zero_number}")

            break
        number += 1

    if not valid_start_number:
        log("No valid start number found", 'error')
        return {"validStartNumber": None, "validFrames": 0, "zeroPageExists": False, "zeroPageUrls": [], "validUrls": []}

    # Step 2: Count valid frames and collect valid URLs
    frame_number = 1  # Start with frame 01
    previous_frame_valid = False  # Track if the previous frame was valid
    current_start_number = valid_start_number  # Start checking from the first valid start number
    consecutive_failures = 0
    max_consecutive_failures = 3  # Stop after this many consecutive failures

    log(f"Starting to collect valid frames from start number: {current_start_number}")

    while True:
        padded_frame_number = str(frame_number).zfill(2)  # Convert 1 → 01, 2 → 02, etc.
        log(f"Checking frame {padded_frame_number} at number {current_start_number}")
        
        current_urls_to_check = [url.replace('*', str(current_start_number)).replace('#', padded_frame_number) 
                                for url in base_urls]

        results = await asyncio.gather(*[check_url(url, session) for url in current_urls_to_check])
        is_valid = any(results)

        if not is_valid:
            consecutive_failures += 1
            log(f"Frame {padded_frame_number} not found (failure {consecutive_failures}/{max_consecutive_failures})")
            
            if consecutive_failures >= max_consecutive_failures:
                log(f"Reached {max_consecutive_failures} consecutive failures, stopping scan")
                break
            
            if previous_frame_valid:
                # Try next frame before giving up
                frame_number += 1
                current_start_number += 1
                continue
            
            # If first frame is invalid and we have consecutive failures, stop
            if frame_number == 1:
                break
        else:
            consecutive_failures = 0  # Reset consecutive failures counter
            log(f"Found valid frame {padded_frame_number}", 'success')
            
            # Collect valid URLs for this frame
            for url, is_valid in zip(current_urls_to_check, results):
                if is_valid:
                    valid_urls.append(url)

            # Update the previous frame validity
            previous_frame_valid = True
            valid_frames += 1  # Increment the valid frame count

        # Move to the next frame and increment the start number
        frame_number += 1
        current_start_number += 1

    log(f"Found a total of {valid_frames} valid frames", 'success')
    return {
        "validStartNumber": valid_start_number, 
        "validFrames": valid_frames, 
        "zeroPageExists": zero_page_exists, 
        "zeroPageUrls": zero_page_urls, 
        "validUrls": valid_urls
    }

# Download images
async def download_images(urls, output_folder, session):
    os.makedirs(output_folder, exist_ok=True)
    
    downloaded_files = []
    
    for i, url in enumerate(urls):
        filename = f"page_{str(i).zfill(3)}.jpg"
        output_path = os.path.join(output_folder, filename)
        
        log(f"Downloading image {i + 1}/{len(urls)}: {url}")
        
        try:
            full_url = url if url.startswith('http') else f"{proxy_url}{url}"
            async with session.get(full_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download: {response.reason}")
                
                image_data = await response.read()
                
                # Convert to JPEG if needed
                if url.endswith('.webp'):
                    img = Image.open(BytesIO(image_data))
                    img = img.convert('RGB')
                    img.save(output_path, 'JPEG')
                else:
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                
                downloaded_files.append(output_path)
                log(f"Downloaded: {filename}", 'success')
        except Exception as error:
            log(f"Failed to download image: {str(error)}", 'error')
    
    return downloaded_files

# Create HTML file
def create_html(series_name, chapter_number, image_urls, output_folder):
    # Sanitize series name and chapter for filename
    sanitized_series_name = re.sub(r'[^a-z0-9]', '_', series_name.lower())
    sanitized_chapter = re.sub(r'[^a-z0-9]', '_', str(chapter_number).lower())
    
    filename = f"{sanitized_series_name}_chapter_{sanitized_chapter}.html"
    output_path = os.path.join(output_folder, filename)
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{series_name} - Chapter {chapter_number}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: #121212;
            color: #E0E0E0;
            font-family: Arial, sans-serif;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #1E1E1E;
            border-radius: 8px;
        }}
        .image-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{series_name}</h1>
            <h2>Chapter {chapter_number}</h2>
        </div>
        <div class="image-container">
"""
    
    # Add images - use local paths relative to HTML file
    for i in range(len(image_urls)):
        filename = f"page_{str(i).zfill(3)}.jpg"
        html_content += f'            <img src="{filename}" alt="Page {i + 1}" loading="lazy">\n'
    
    html_content += """
        </div>
    </div>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log(f"Created HTML file: {filename}", 'success')
    
    return output_path

# Main process function
async def process_images():
    global is_scanning, current_start_number
    
    if is_scanning:
        log("Already scanning, will check again later", 'info')
        return
    
    is_scanning = True
    
    try:
        # Load previous state
        state = load_state()
        
        async with aiohttp.ClientSession() as session:
            log(f"Starting scan from number {current_start_number}")
            result = await find_valid_start_and_frames(current_start_number, session)
            
            if not result["validStartNumber"]:
                log("No valid images found in this scan", 'info')
                is_scanning = False
                return
            
            # Get all valid URLs
            all_urls = result["zeroPageUrls"] + result["validUrls"] if result["zeroPageExists"] else result["validUrls"]
            
            log(f"Found {len(all_urls)} valid images", 'success')
            
            # Extract series name and chapter
            series_name = "Unknown Series"
            chapter_number = "unknown"
            
            if result["zeroPageExists"] and result["zeroPageUrls"]:
                try:
                    extracted_info = await extract_text_from_image(result["zeroPageUrls"][0], session)
                    series_name = extracted_info["seriesName"] or series_name
                    chapter_number = extracted_info["chapterNumber"] or chapter_number
                    log(f'Extracted series: "{series_name}", chapter: {chapter_number}', 'success')
                except Exception as error:
                    log(f"Failed to extract info from zero page: {str(error)}", 'error')
            
            # Create folder for series
            series_folder = os.path.join(output_dir, re.sub(r'[^a-z0-9]', '_', series_name.lower()))
            os.makedirs(series_folder, exist_ok=True)
            
            # Create chapter folder
            chapter_folder = os.path.join(series_folder, f"chapter_{re.sub(r'[^a-z0-9]', '_', str(chapter_number).lower())}")
            os.makedirs(chapter_folder, exist_ok=True)
            
            # Check if we've already downloaded this chapter
            chapter_key = f"{series_name}_{chapter_number}"
            if "downloadedChapters" in state and chapter_key in state["downloadedChapters"]:
                log(f"Chapter {chapter_number} of {series_name} already downloaded, skipping", 'info')
            else:
                # Download images
                log(f"Downloading chapter {chapter_number} of {series_name}")
                downloaded_files = await download_images(all_urls, chapter_folder, session)
                
                if downloaded_files:
                    # Create HTML
                    html_path = create_html(series_name, chapter_number, all_urls, chapter_folder)
                    
                    # Mark as downloaded
                    if "downloadedChapters" not in state:
                        state["downloadedChapters"] = {}
                    
                    state["downloadedChapters"][chapter_key] = {
                        "timestamp": datetime.now().isoformat(),
                        "imageCount": len(downloaded_files),
                        "path": chapter_folder
                    }
                    
                    log(f"Successfully processed chapter {chapter_number} of {series_name}", 'success')
            
            # Update the next starting number to search from
            # The next start number should be the last valid start number plus the number of valid frames
            next_start_number = result["validStartNumber"] + result["validFrames"]
            log(f"Next scan will start from number {next_start_number}")
            
            # Update state
            state["lastStartNumber"] = next_start_number
            save_state(state)
            
            # Update global variable for next scan
            current_start_number = next_start_number
            
    except Exception as error:
        log(f"Error in processing: {str(error)}", 'error')
    finally:
        is_scanning = False

# Main execution
async def main():
    log('Webtoon Downloader Backend Started', 'info')
    
    # Initial run
    await process_images()
    
    while True:
        log('Running scheduled check')
        await process_images()
        # Wait for 5 minutes
        await asyncio.sleep(5 * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Process interrupted by user", 'info')
    except Exception as error:
        log(f"Fatal error: {str(error)}", 'error')
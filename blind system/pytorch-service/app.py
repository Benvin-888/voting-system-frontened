import io
import torch
import torchaudio
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging
import numpy as np
import tempfile
import os
import re
import librosa
import soundfile as sf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Use a more accurate model fine-tuned for speech recognition
MODEL_NAME = "facebook/wav2vec2-large-960h"  # More accurate than base model
logger.info(f"Loading model: {MODEL_NAME}...")

try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    # Fallback to base model if large model fails
    MODEL_NAME = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    logger.info("Loaded base model as fallback")

# Voting-specific vocabulary for better recognition
VOTING_KEYWORDS = {
    'login': ['login', 'log in', 'sign in'],
    'governor': ['governor', 'gov', 'governa'],
    'women_rep': ['women representative', 'women rep', 'woman rep'],
    'mp': ['mp', 'member of parliament', 'M P'],
    'mca': ['mca', 'member of county assembly', 'M C A'],
    'confirm': ['confirm', 'yes', 'submit', 'cast'],
    'change': ['change', 'no', 'edit', 'modify'],
    'repeat': ['repeat', 'again', 'listen again'],
    'help': ['help', 'assist', 'instructions'],
    'start': ['start', 'begin', 'vote']
}

def enhance_transcription(text):
    """Post-process transcription to better match voting commands"""
    if not text:
        return ""
    
    text = text.lower().strip()
    
    # Remove extra spaces and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    
    # Check for voting keywords
    for command, patterns in VOTING_KEYWORDS.items():
        for pattern in patterns:
            if pattern in text:
                return command
    
    # Extract numbers (for voting number)
    numbers = re.findall(r'\d+', text)
    if numbers and len(''.join(numbers)) >= 10:
        return ''.join(numbers)[:10]
    
    return text

@app.route('/asr', methods=['POST'])
def asr():
    """Convert uploaded audio to text with voting-specific enhancements"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        if not audio_bytes:
            return jsonify({'error': 'Empty audio file'}), 400

        logger.info(f"Received audio: {len(audio_bytes)} bytes")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name
        
        try:
            # Load audio with librosa (better format support)
            waveform, sample_rate = librosa.load(temp_path, sr=16000, mono=True)
            
            # Convert to tensor
            waveform = torch.from_numpy(waveform).float()
            
            logger.info(f"Audio loaded: sample_rate={sample_rate}, waveform shape={waveform.shape}")
            
            # Ensure waveform is 2D [1, samples] if needed
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            # Prepare for model
            input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            
            # Run inference
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            # Enhance transcription for voting commands
            enhanced_text = enhance_transcription(transcription)
            
            logger.info(f"Raw transcription: {transcription}")
            logger.info(f"Enhanced text: {enhanced_text}")
            
            return jsonify({
                'text': enhanced_text,
                'raw_text': transcription,
                'success': True
            })
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
    except Exception as e:
        logger.error(f"Error in ASR: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/test', methods=['GET'])
def test():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voting ASR Test</title>
        <style>
            body { font-family: Arial; padding: 20px; max-width: 600px; margin: 0 auto; }
            button { padding: 15px 30px; font-size: 18px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #45a049; }
            button.stop { background: #f44336; }
            button.stop:hover { background: #da190b; }
            #result { margin-top: 20px; padding: 20px; border: 2px solid #ddd; border-radius: 5px; min-height: 50px; font-size: 18px; }
            #status { margin-top: 10px; color: #666; }
            .error { color: red; }
            .success { color: green; }
            .command-list { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
            .command-item { display: inline-block; margin: 5px; padding: 5px 10px; background: #e0e0e0; border-radius: 15px; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>🗳️ Voting Speech Recognition Test</h1>
        <p>Click and speak voting commands (login, governor, mp, mca, confirm, change, help, start)</p>
        
        <div class="command-list">
            <strong>Supported Commands:</strong>
            <span class="command-item">login</span>
            <span class="command-item">governor</span>
            <span class="command-item">women_rep</span>
            <span class="command-item">mp</span>
            <span class="command-item">mca</span>
            <span class="command-item">confirm</span>
            <span class="command-item">change</span>
            <span class="command-item">repeat</span>
            <span class="command-item">help</span>
            <span class="command-item">start</span>
        </div>
        
        <button id="record">Start Recording (3s)</button>
        <button id="record5">Start Recording (5s)</button>
        <p id="status">Ready</p>
        <div id="result"></div>
        
        <script>
            let mediaStream = null;
            let audioContext = null;
            let source = null;
            let processor = null;
            let chunks = [];
            let recordingTimeout = null;
            
            // Convert float32 array to WAV format
            function floatToWav(audioData, sampleRate) {
                const buffer = new ArrayBuffer(44 + audioData.length * 2);
                const view = new DataView(buffer);
                
                // RIFF chunk descriptor
                writeString(view, 0, 'RIFF');
                view.setUint32(4, 36 + audioData.length * 2, true);
                writeString(view, 8, 'WAVE');
                
                // fmt subchunk
                writeString(view, 12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true); // PCM format
                view.setUint16(22, 1, true); // Mono
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true); // byte rate
                view.setUint16(32, 2, true); // block align
                view.setUint16(34, 16, true); // bits per sample
                
                // data subchunk
                writeString(view, 36, 'data');
                view.setUint32(40, audioData.length * 2, true);
                
                // Write audio data (convert float32 to int16)
                floatTo16BitPCM(view, 44, audioData);
                
                return new Blob([buffer], { type: 'audio/wav' });
            }
            
            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }
            
            function floatTo16BitPCM(view, offset, input) {
                for (let i = 0; i < input.length; i++, offset += 2) {
                    let s = Math.max(-1, Math.min(1, input[i]));
                    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                }
            }
            
            async function startRecording(duration) {
                const status = document.getElementById('status');
                const result = document.getElementById('result');
                
                try {
                    // Stop any existing recording
                    stopRecording();
                    
                    status.textContent = 'Requesting microphone access...';
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    // Use AudioContext for better control
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    source = audioContext.createMediaStreamSource(mediaStream);
                    processor = audioContext.createScriptProcessor(4096, 1, 1);
                    
                    chunks = [];
                    
                    processor.onaudioprocess = (e) => {
                        const inputData = e.inputBuffer.getChannelData(0);
                        chunks.push(new Float32Array(inputData));
                    };
                    
                    source.connect(processor);
                    processor.connect(audioContext.destination);
                    
                    status.textContent = `Recording... (${duration} seconds)`;
                    result.innerHTML = '';
                    
                    // Stop after duration
                    recordingTimeout = setTimeout(async () => {
                        await processRecording(duration);
                    }, duration * 1000);
                    
                } catch (err) {
                    status.textContent = 'Error: ' + err.message;
                    result.innerHTML = 'Microphone access denied or not available';
                    result.className = 'error';
                }
            }
            
            function stopRecording() {
                if (recordingTimeout) {
                    clearTimeout(recordingTimeout);
                    recordingTimeout = null;
                }
                
                if (processor) {
                    try { processor.disconnect(); } catch(e) {}
                    processor = null;
                }
                
                if (source) {
                    try { source.disconnect(); } catch(e) {}
                    source = null;
                }
                
                if (audioContext) {
                    try { audioContext.close(); } catch(e) {}
                    audioContext = null;
                }
                
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
            }
            
            async function processRecording(duration) {
                const status = document.getElementById('status');
                const result = document.getElementById('result');
                
                if (source) {
                    source.disconnect();
                }
                if (processor) {
                    processor.disconnect();
                }
                
                // Combine all chunks
                const totalLength = chunks.reduce((acc, val) => acc + val.length, 0);
                const combined = new Float32Array(totalLength);
                let offset = 0;
                for (const chunk of chunks) {
                    combined.set(chunk, offset);
                    offset += chunk.length;
                }
                
                status.textContent = 'Converting to WAV...';
                
                // Convert to WAV
                const wavBlob = floatToWav(combined, audioContext ? audioContext.sampleRate : 16000);
                
                status.textContent = 'Sending to server...';
                
                // Send to server
                const formData = new FormData();
                formData.append('audio', wavBlob, 'recording.wav');
                
                try {
                    const res = await fetch('/asr', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await res.json();
                    
                    if (data.success) {
                        let displayText = '<strong>You said:</strong> "' + data.text + '"';
                        if (data.raw_text && data.raw_text !== data.text) {
                            displayText += '<br><small>Raw: "' + data.raw_text + '"</small>';
                        }
                        result.innerHTML = displayText;
                        result.className = 'success';
                        status.textContent = 'Done!';
                    } else {
                        result.innerHTML = '<strong>Error:</strong> ' + (data.error || 'Unknown error');
                        result.className = 'error';
                        status.textContent = 'Failed';
                    }
                } catch (err) {
                    result.innerHTML = '<strong>Network Error:</strong> ' + err.message;
                    result.className = 'error';
                    status.textContent = 'Failed';
                } finally {
                    stopRecording();
                }
            }

            document.getElementById('record').onclick = function() {
                startRecording(3);
            };
            
            document.getElementById('record5').onclick = function() {
                startRecording(5);
            };
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    logger.info("Starting Voting ASR server on http://0.0.0.0:5005")
    logger.info("Test the service at: http://localhost:5005/test")
    app.run(host='0.0.0.0', port=5005, debug=True, threaded=True)
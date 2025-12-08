from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import eventlet
from main2 import process_stream

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

@app.route('/')
def index():
    return "YOLO Stream Processor"

@app.route('/start_stream', methods=['POST'])
def start_stream():
    data = request.get_json()
    stream_url = data.get('stream_url')
    session_id = request.sid # Get session ID for specific client emission

    if not stream_url:
        return jsonify({"error": "No stream_url provided"}), 400

    # Start the stream processing in a background thread
    eventlet.spawn(process_stream, stream_url, lambda event, data: socketio.emit(event, data, room=session_id))

    return jsonify({"message": f"Stream processing started for {stream_url}"}), 200

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

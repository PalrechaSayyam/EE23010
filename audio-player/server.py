from flask import Flask, send_file

app = Flask(__name__)

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_path = f'/storage/self/primary/Download/EE23010/audio-player/audio/{filename}'
    return send_file(audio_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



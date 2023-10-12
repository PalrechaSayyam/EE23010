from flask import Flask, send_file

app = Flask(__name__)

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_path = f'/storage/self/primary/Download/EE23010/audio-player/audio/{filename}'
    return send_file(audio_path)

@app.route('/pages/<filename>')
def serve_html(filename):
    html_path = f'/storage/self/primary/Download/EE23010/audio-player/src/pages/{filename}'
    return send_file(html_path)

@app.route('/components/<filename>')
def serve_css(filename):
    css_path = f'/storage/self/primary/Download/EE23010/audio-player/src/components/{filename}'  # Update the path to your CSS file
    return send_file(css_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



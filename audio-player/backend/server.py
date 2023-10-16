from flask import Flask, send_file
import random
import os

app = Flask(__name__)

audio_directory = '/storage/self/primary/Download/EE23010/audio-player/audio'

@app.route('/play-random-audio')
def play_random():
    audio_files = os.listdir(audio_directory)
    uni_dist = [random.uniform(0,1) for _ in range(len(audio_files))]
    indices = [int(rand_no*len(audio_files)) for rand_no in uni_dist]
    random_file = audio_files[indices[0]]
    audio_path = os.path.join(audio_directory, random_file)
    return render_template('front.html')

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_path = f'/storage/self/primary/Download/EE23010/audio-player/audio/{filename}'
    return send_file(audio_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



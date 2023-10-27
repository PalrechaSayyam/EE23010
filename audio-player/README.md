# Audio-Player

This directory contains all the necessary modules to develop a custom audio player on Termux!!

## setup

### Termux Installation and setup

You can look into the link below on installing and setting up Termux on your smartphone.

https://github.com/gadepall/fwc-1#fwc-1

Follow till step three of [Installing and Setting up Ubuntu on Termux
](https://github.com/gadepall/fwc-1#installing-and-setting-up-ubuntu-on-termux), and you are ready with the set-up.

### Audio Player setup

#### Virtual Environment setup

Once your Termux set-up is done, go to the terminal and run the following commands to install ```venv```:
```bash
sudo apt update
sudo apt install python3-venv
```
After successfully installing  ```venv``` run this command to create a virtual environment:
```bash
python3 -m venv flaskenv
```

#### Clone this repository

Run the following command to clone this repo:

```bash
git clone https://github.com/PalrechaSayyam/EE23010.git
```
Activate the virtual environment:
```bash
source flaskenv/bin/activate
```
#### Server setup

Once cloned, head to the audio-player directory using the command:
```bash
cd audio-player
```
Run the following command to install all the required dependencies:
```bash
pip install -r requirements.txt
```
Head to the ```backend``` directory to start the server:
```bash
cd backend
python3 server.py
```

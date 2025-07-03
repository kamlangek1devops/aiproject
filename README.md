# AI Project

AI Project for Assessment II

### Model Train - Google Colab
- https://colab.research.google.com/drive/1Qrgnl2S7rT55vAkuqbhibPrEa6Vzolc4#scrollTo=3K8pB1iLxUu8

### Prerequisite

Create a virtual environment

```bash
python3 -m venv .venv
```

Activate the virtual environment

```bash
source .venv/bin/activate  # Linux/Mac
source .venv/Scripts/activate  # Windows
```

Install the required packages

```bash
pip install ultralytics
pip install "numpy<2"
pip install pyqt6
```

Or you can install from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Deploy Method

- Image: `python deploy.py --model my_model.pt --source metal7.jpg`
- Camera Feed: `python deploy.py --model my_model.pt --source usb0 --resolution 1280x720`
- GUI: `python MainApp.py --model my_model.pt`
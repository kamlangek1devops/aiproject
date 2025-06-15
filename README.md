# AI Project
Project AI for assessment II

### Deploy Model on Local
#### Prerequisite
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install ultralytics`
- `pip install "numpy<2"`
- `pip install pyqt6`

#### Deploy Method:
- Image: `python deploy.py --model my_model.pt --source metal7.jpg`
- Camerafeed: `python deploy.py --model my_model.pt --source usb0 --resolution 1280x720`
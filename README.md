# Traffic-Intersection-Monitor

## Description

A mini project on vehicle counting and direction classification, built on YOLOv8 and ByteTrack. It watches a video of an intersection, that tracks each vehicle with a persistent ID and reports:

- **Per class counts** of vehicles passing through a defined counting zone (cars, buses, trucks, motorcycles), each tracked ID is counted once, no matter how many framesit lingers in the zone.
- **Direction breakdown** (straight/left/right) of vehicles passing through a second zone placed over an intersection, based on which edge of that zone a vehicle enters from versus which edge it exits from.

Two seperate code for different angle view of the intersection:

- `Traffic_Monitoring_Top_View.py` code for **Top-down** or **aerial footage** (e.g: drone shots, traffic camera, high mounted CCTC) of the intersection.
- `Traffic_Monitoring_Side_View.py` code for **Side-on** or **Street-level footage** (e.g: dash cam, phone footage shot from the ground) of the intersection.

## Installation

This Script requires Python 3.8 or newer installed.

It is recommended to create a virtual environment, before installing dependancies, so that the project dependancies stay isolated from other Python projects.

#### 1. Create Virtual Environment

Navigate to project folder and create a virtual environment

`python -m venv venv`

This creates a folder called venv that contains an isolated Python environment for the project.

#### 2. Activate Virtual Environment

On macOS/Linux:

`source venv/bin/activate`

On Windows:

`venv\Scripts\activate.bat`

Once activated, terminal should show (venv) at the beginning of the command line.

#### 3. Install Dependancies

With the virtual environment active, install required libraries:

`pip install opencv-python ultralytics matplotlib`

Or if using a requirements file:

`pip install -r requirements.txt`

The two main libraries used in the project are:

- **Opencv-python** handles reading, processing and displaying images and video
- **Ultralytics** gives the YOLOv8 object detection model that is ready to use.

#### 4. Run the application

`python Traffic_Monitoring_Top_View.py`

Or

`python Traffic_Monitoring_Side_View.py`

Press `q` in the video window to stop early. Final and direction counts are printed to the console when the video ends or you quit.

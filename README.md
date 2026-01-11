# Chest X-Ray Search System

This is the code repository for the chest x-ray search system.

## Setting up

This project has been tested on Python 3.13.

First, install the dependencies needed to run this app:

```
python -m venv .venv

# macOS/Linux/other *nix
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

You will also need to download the images from the dataset used ([CXR8](https://nihcc.app.box.com/v/ChestXray-NIHCC)).
For your convenience, a [download_images.sh](./download_images.sh) script is included to
automatically download and extract everything.

## Running

Inside the virtual environment:

```
streamlit run app.py
```

## Simple attempt at a video compression algorithm

### Setup
```bash
git clone https://github.com/CrestyDY/vid-compression.git

cd vid-compression
```

Make a virtual env if you want:
```bash
python -m venv .venv
```
Activate it and install dependencies

```bash
pip install uv
uv sync
```

### Structure

All source files are inside of src. Inside of src, you will also find:
  - models: Contains all pydantic models for all the objects that are used in the project.
  - modes: Contains all the different modes that are used to represent a new frame
  - utils: Contains all the utilities that are used in the project, each inside their own files


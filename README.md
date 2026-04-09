# DeepFake Frame Capture

DeepFrameC is a DeepFake detection Engine built to detect both audio and video deepfakes with confidence scores.

## Installation and Setup

### Prerequisites

Ensure you have Python 3.9+ and Node.js installed on your system.

### 1. Install Python Dependencies

It is recommended to use a virtual environment before installing the dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required Python packages
pip install -r backend/requirements.txt
```

### 2. Build the Frontend

Navigate to the frontend directory inside the backend and build the React application:

```bash
cd backend/frontend
npm install
npm run build
cd ../..
```

## Running the Application

To start the FastAPI backend server, run the following commands from the root directory:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start up and you can access the application at `http://localhost:8000`. 
On the first startup, it will download the necessary model weights.

# scripts/api_server.py

"""
A simple Flask API server to trigger the execution of our project's agents.
This server runs on the host machine and listens for HTTP requests from n8n
or any other client, bridging the gap between the Docker container and the
host's Conda environment.
"""

from flask import Flask, jsonify
import subprocess
import os

# --- Configuration ---
app = Flask(__name__)
# Get the project root directory relative to this script's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_1_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'agent_data_processing.py')
AGENT_2_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'agent_graph_construction.py')
CONDA_ENV_NAME = 'pharmagraph-py311'

# --- Helper Function to Run a Script ---
def run_agent_script(script_path):
    """A helper function to run a given Python script in our conda environment."""
    command = ['conda', 'run', '-n', CONDA_ENV_NAME, 'python', script_path]
    try:
        print(f"--- Running command: {' '.join(command)} ---")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # This will raise an exception if the script returns a non-zero exit code
            shell=True   # Important for making 'conda' accessible on Windows
        )
        print("--- Script executed successfully ---")
        return {"status": "success", "output": result.stdout}, 200
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Script failed to execute ---")
        print(f"Stderr: {e.stderr}")
        return {"status": "error", "error_message": e.stderr}, 500
    except Exception as e:
        print(f"--- ERROR: An unexpected error occurred ---")
        print(str(e))
        return {"status": "error", "error_message": str(e)}, 500

# --- API Endpoints ---
@app.route('/run_agent_1', methods=['POST'])
def trigger_agent_1():
    """API endpoint to trigger the data processing agent."""
    print("\nReceived request to run Agent 1: Data Processing...")
    response, status_code = run_agent_script(AGENT_1_PATH)
    return jsonify(response), status_code

@app.route('/run_agent_2', methods=['POST'])
def trigger_agent_2():
    """API endpoint to trigger the graph construction agent."""
    print("\nReceived request to run Agent 2: Graph Construction...")
    response, status_code = run_agent_script(AGENT_2_PATH)
    return jsonify(response), status_code

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Flask API server for PharmacoGraph-Agent ---")
    print("Listening for requests on http://localhost:8000")
    # Use host='0.0.0.0' to make it accessible from outside the container
    app.run(host='0.0.0.0', port=8000)
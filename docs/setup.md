# Setup Guide

Follow this guide to get Reachy Nova running on your Reachy Mini robot or development machine.

## Prerequisites

-   **Operating System**: Linux (Ubuntu 22.04+ recommended) or macOS.
-   **Python**: Version 3.12 or newer.
-   **UV**: The `uv` package manager (recommended) or robust `pip`/`venv` setup.
-   **AWS Account**: Access to Amazon Bedrock with the following models enabled:
    -   `amazon.nova-sonic-v1:0` (us-east-1)
    -   `us.amazon.nova-pro-v1:0` (us-east-1)
    -   Access to Nova Act execution role if running browser automation.

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/pollen-robotics/reachy_nova.git
    cd reachy_nova
    ```

2.  **Install Dependencies**:
    We recommend using `uv` for fast dependency resolution.
    ```bash
    uv sync
    ```
    This will install `reachy-mini`, `boto3`, `nova-act`, `opencv-python`, and other required packages.

3.  **Configure Environment**:
    Create a `.env` file from the sample:
    ```bash
    cp .env.sample .env
    ```

    Edit `.env` with your AWS credentials:
    ```ini
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    AWS_DEFAULT_REGION=us-east-1
    ```

## Running the Application

### Development Mode

Run the application directly:
```bash
uv run python -m reachy_nova.main
```

The application will:
1.  Connect to Reachy Mini (or start a mock if no robot is found).
2.  Start the internal web server at `http://localhost:8042`.
3.  Begin listening for voice commands.

### Hardware Setup (Reachy Mini)

Ensure your Reachy Mini is connected via USB or properly networked. The `reachy-mini` library handles device discovery.

-   **Camera**: Ensure the camera is accessible (check `/dev/video*` permissions on Linux).
-   **Microphone**: Ensure the microphone is set as the default input or properly selected by `pyaudio`.
-   **Speakers**: Ensure audio output is configured.

## Troubleshooting

### Voice Not Working
-   Check AWS credentials in `.env`.
-   Verify `amazon.nova-sonic-v1:0` access in AWS Bedrock console.
-   Check microphone permissions.

### Vision Not Working
-   Verify camera access (`cv2.VideoCapture`).
-   Check `us.amazon.nova-pro-v1:0` access in AWS Bedrock.

### Browser Automation Fails
-   Ensure `nova-act` is installed correctly.
-   Check if Playwright browsers are installed:
    ```bash
    playwright install chromium
    ```

### Tracking Issues
-   If face tracking is slow, ensure you are running on a machine with decent CPU or GPU.
-   YOLOv8n is used by default for speed.

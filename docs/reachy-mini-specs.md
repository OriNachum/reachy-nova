# Reachy Mini Wireless: Technical Overview

Reachy Mini Wireless is an expressive, untethered desktop robot designed for Human-Robot Interaction (HRI) and social robotics research. It is a compact, 9-DOF (Degrees of Freedom) platform that integrates onboard compute, vision, and audio processing into a mobile form factor.

---

## Technical Specifications

### Core Compute & Connectivity
* **Processor:** Raspberry Pi Compute Module 4 (CM4104016)
* **RAM:** 4GB LPDDR4-3200 SDRAM
* **Storage:** 16GB eMMC Flash
* **Wireless:** 2.4 GHz, 5.0 GHz IEEE 802.11 b/g/n/ac wireless; Bluetooth 5.0, BLE
* **Antenna:** 2.79 dBi patch antenna

### Power System
* **Battery Type:** LiFePO4 (Lithium Iron Phosphate)
* **Capacity:** 2000 mAh / 6.4 V (12.8 Wh)
* **Input Voltage:** 6.8V – 7.6V
* **Charging:** USB-C 

### Kinematics (9 Degrees of Freedom)
* **Head (6-DOF):** Stewart platform configuration allowing 3 rotations (pitch, roll, yaw) and 3 translations (x, y, z).
* **Body (1-DOF):** 360-degree continuous yaw rotation.
* **Antennas (2-DOF):** Independent movement for expressive signaling (1 DOF per antenna).

### Vision & Audio
* **Camera:** Sony IMX708 (Raspberry Pi Camera Module 3); 12MP, 120° wide-angle FoV, autofocus support.
* **Video Output:** H.264 hardware encoding, streaming via WebRTC (up to 1080p @ 30fps).
* **Microphones:** 4x PDM MEMS digital microphones (XMOS XVF3800 processor).
* **Speaker:** 1x 5W (4Ω) internal speaker.
* **Sensors:** Integrated 6-axis IMU (Accelerometer/Gyroscope).

---

## Software & Integration
* **Operating System:** Linux-based (Raspberry Pi OS).
* **SDK:** Python >= 3.10 support.
* **Communication:** Local daemon managing motor control and sensor telemetry; accessible via HTTP REST API and WebSocket.
* **Media Pipeline:** GStreamer-based video and audio streaming with Opus compression.
* **AI Compatibility:** Supports integration with Hugging Face, OpenAI, and custom local/remote inference engines.

---

## Physical Characteristics & Design
* **Dimensions:** 300 x 200 x 155 mm (Fully extended).
* **Weight:** 1.475 kg.
* **Materials:** ABS, Polycarbonate, Aluminium, and Steel.
* **Aesthetic:** Industrial-realistic torso design featuring a single central lens behind a protective dome and dual mechanical antennas on the head for non-verbal communication.

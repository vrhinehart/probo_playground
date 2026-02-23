# Probabilistic Robotics Playground: Linear and Extended Kalman Filtering

Requires uv: https://docs.astral.sh/uv/   
To run: 
`uv run -m probo_sim.main`

<img width="500" alt="kf_gps" src="https://github.com/user-attachments/assets/801b9f6f-6be0-4745-a1c1-10f9685b4a5a" />

This is a regular linear Kalman filter informed by wheel speed sensors and GPS. The estimate (red) deviates from the true course (green) but is drawn back in by the GPS (black points). 

<img width="500" alt="ekf_gps_lm" src="https://github.com/user-attachments/assets/fbd59c90-f009-4d8d-a6a4-a5adb6c185d5" />

This is an extended Kalman filter informed by wheel speed, GPS, and landmark pingers. The estimate (red) tracks the true course (green) much more closely. GPS readings are shown in bolack, and the blue stars represent landmarks. Yellow rays represent landmark pinger readings.

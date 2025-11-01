AI Feature Extraction From Drone Orthophotos

This project uses a CNN-based image segmentation model to detect features like rooftops, roads, and waterbodies from drone orthophotos. It includes an interactive web dashboard built with Streamlit for uploading images and viewing the results.

Tech Stack
Python
Streamlit (for the web dashboard)
TensorFlow (for the CNN model)
Scikit-learn (for model processing and metrics)
Pandas (for data handling)
OpenCV (for image processing)

Installation
To run this project locally, follow these steps:

1.Clone the repository:
    git clone [https://github.com/ghoshrahul10/AI-Feature-Extraction-From-Drone-Orthophotos.git](https://github.com/ghoshrahul10/AI-Feature-Extraction-From-Drone-Orthophotos.git)
    cd AI-Feature-Extraction-From-Drone-Orthophotos

2. Create and activate a virtual environment:
    Create the environment
    python -m venv venv
    
    Activate on Windows (PowerShell)
    .\venv\Scripts\Activate.ps1

3. Install the required dependencies:
    pip install -r requirements.txt

---

How to Run
With your virtual environment active, run the Streamlit app:

```bash
streamlit run Societal.py
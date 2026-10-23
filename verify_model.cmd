@echo off
cd /d c:\Users\USER\Desktop\bridgesign-main
.\.venv\Scripts\python.exe -c "import app; app._load_inference_modules(); print('ready', app._inference_status.get('ready')); print('model', type(app._classifier).__name__ if app._classifier else None); print('A', app._classifier.predict(app._SANITY_SAMPLES['A']))"

#import statements
import sounddevice as sd
import numpy as np
import openwakeword
from openwakeword.model import Model

# One-time download of the pre-trained models to test
openwakeword.utils.download_models()

# instantiate the model
model = Model(wakeword_models=["hey_jarvis_v0.1"])              # Model() could be left empty to load all models

# cooldown period to prevent multiple detections in quick succession
cooldown_frames = 20
current_cooldown = 0

with sd.InputStream(channels=1, samplerate=16000, blocksize=1280, dtype='int16') as stream:
    print("Listening for wake word...")

    while True:
        data, overflowed = stream.read(1280)

        if overflowed:
            print("Warning: Input overflow detected, wake word detection may be unreliable.")

        if current_cooldown > 0:
            current_cooldown -= 1
            continue                 # skip processing during cooldown  

        frame = data.flatten()
        prediction = model.predict(frame)

        if prediction['hey_jarvis_v0.1'] > 0.9:  # threshold for wake word detection
            print("At your service, lord!")
            current_cooldown = cooldown_frames
            model.reset()  # reset model state after detection to prevent multiple triggers
"""Per-robot settings. Copy to rover/local_config.py on each robot and edit.
Anything in rover/config.py can be overridden here; everything else is shared code.

Below: the SECOND rover (different motor wiring, only a front and a rear ultrasonic sensor).
"""

ROBOT_NAME = "Nova"                  # its name is also its wake word ("Nova, what's...")
LINK_KEY = "pick-any-secret"         # same on both robots

# its own voice, so the audience can tell the two robots apart (Lyra keeps the default lessac voice)
TTS_VOICE = "models/piper/en_US-hfc_female-medium.onnx"

# the other robot
PEER_NAME = "Lyra"
PEER_HOST = "maximus.local"          # Lyra's hostname (run `hostname` on it) or its IP address

# --- motors (L298N, BCM pin numbers) -- set to this robot's wiring
LEFT_IN1, LEFT_IN2, LEFT_EN = 17, 27, 12
RIGHT_IN3, RIGHT_IN4, RIGHT_EN = 22, 23, 13
LEFT_INVERTED, RIGHT_INVERTED = False, False
MIN_DUTY = 0.35
METRES_PER_SEC_AT_FULL = 0.5         # calibrate: drive 1 s at full speed, measure
DEGREES_PER_SEC_AT_FULL = 180.0      # calibrate: spin 1 s at full speed, measure

# --- only two ultrasonic sensors: (trigger, echo)
ULTRASONIC = {"front": (19, 26), "rear": (24, 25)}
ULTRASONIC_ANGLES = {"front": 0, "rear": 180}

# Bluetooth speaker of this robot (None = PipeWire default device)
BT_AUDIO_MAC = None

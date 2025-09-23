import os

# --- Directory Paths ---
# Assuming the script runs from the project root
# Adjust these paths according to your actual setup

# Base directory where the zipped/cloned project resides
# This is the parent directory of folders like 'Dương', 'Giang', etc.
"""
    Old configs that we do not use anymore: PROJECT_DATA_ROOT, OUTPUT_TXT_ROOT, OUTPUT_TFIDF_ROOT. New configs are below.
"""
PROJECT_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Adjust if needed

# Output directory for the converted .txt files
OUTPUT_TXT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output_txt'))
OUTPUT_TFIDF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output_tfidf'))

# --- File Extensions ---
PDF_EXTENSION = '.pdf'
TEXT_EXTENSION = '.txt'
TFIDF_CSV_FILE = 'additional_tfidf_scores.csv'

CLIMATE_RISK_KEYWORDS_RAW = [
    "adaptive capacity", "air burst", "airburst", "airbursts", "apocalypse",
    "apocalypses", "ash fall", "ashfall", "avalanche", "avalanches", "blizzard",
    "blizzards", "calamity", "camilities", "cataclysm", "cataclysms", "climate change",
    "climate warms", "coastal erosion", "cold wave", "cold waves", "coping capacity",
    "cyclone", "cyclones", "debacle", "debacles", "derecho", "derechos", "disaster",
    "disaster risk", "disaster risk management", "disaster risks", "disasters", "drought",
    "droughts", "earthquake", "earthquakes", "El Nino", "extreme heat", "extreme rain",
    "extreme rainfall", "extreme rains", "extreme temperature", "extreme temperatures",
    "extreme weather", "firestorm", "firestorms", "flood", "flooded", "flooding",
    "floodings", "floods", "fog", "fogs", "forest fire", "forest fires", "freeze",
    "freezes", "freezing", "gale", "gales", "geophysical hazard", "geophysical hazards",
    "global warming", "hail", "hailstorm", "hailstorms", "hazard mitigation", "heat wave",
    "heat waves", "heavy rainfall", "heavy snow", "high wind", "high winds", "hurricane",
    "hurricanes", "hydrological hazard", "hydrological hazards", "La Nina", "landfall",
    "landfalls", "landslide", "landslides", "lava flow", "lava flows", "microburst",
    "microbursts", "mudslide", "mudslides", "natural hazard", "natural hazards",
    "rainfall", "rainfalls", "rainstorm", "rainstorms", "rock fall", "rock-fall",
    "rockfall", "rockfalls", "rogue wave", "rogue waves", "seiche", "seiches",
    "severe winter condition", "severe winter conditions", "snow", "snowstorm",
    "snowstorms", "storm", "storms", "thunderstorm", "thunderstorms", "tornado",
    "tornadoes", "tremor", "tsunami", "tsunamis", "twister", "typhoon", "typhoons",
    "volcanic", "volcano", "volcanos", "whirlpool", "whirlpools", "wildfire",
    "wildfires", "wind", "winds", "windstorm", "winterstorm", "winterstorms"
]

CLIMATE_RISK_KEYWORDS = [keyword.lower() for keyword in CLIMATE_RISK_KEYWORDS_RAW]
EXISTING_VARS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'existing_vars.csv') 
FINAL_DATASET_FILE = os.path.join(os.path.dirname(__file__), 'data', 'final_dataset_for_modeling.csv')

NEW_DATASET_FILE = "/mnt/e/NEUConference/Input_Variable"
NEW_DATASET_OUTPUT = "/mnt/e/NEUConference/ClimateRisk/output_txt/Additional"
NEW_FINAL_CSV_PATH = "/mnt/e/NEUConference/ClimateRisk/output_tfidf/Combined_Company_Data_2022_2024_Final2.csv"

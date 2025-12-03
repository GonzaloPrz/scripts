import pandas as pd
from pathlib import Path
from sentiment_analysis import create_analyzer

analyzer = create_analyzer(language="es",task="emotion")


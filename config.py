"""EDIT THIS FILE FIRST. All scripts read their paths and sizes from here."""
# Folder that directly contains blues/, classical/, ... (the "genres_original" folder)
GTZAN_ROOT = r"C:\Users\admin\Documents\Genre"
# Folder that contains BOTH fma_small/ (the mp3s) and fma_metadata/ (tracks.csv)
FMA_ROOT = r"C:\Users\admin\Documents\Genre"
# Clips kept per genre (balanced). GTZAN has 100/genre, FMA-small has 1000/genre.
# FMA at 400/genre = 3,200 clips = about 4.2 GB of cache (int16). Lower it if disk is tight.
PER_GENRE = {"gtzan": 100, "fma": 400}
# Filenames to drop from GTZAN (Sturm 2013 fault list), e.g. ["jazz.00054.wav"]. Leave [] if unsure.
GTZAN_DROP = []
CACHE_DIR = "cache"
RESULTS_DIR = "results"

import pandas as pd
import numpy as np
from utils import *

dict_df = load_data()

print_stats(dict_df['p1'], 'p1')
print_stats(dict_df['p2'], 'p2')
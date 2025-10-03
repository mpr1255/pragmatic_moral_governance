#%%
import pandas
import random
import os
import sys
import pprint
here = os.path.dirname(sys.prefix)
os.chdir(here)

df = pandas.read_csv(f"{here}/task_cleanup/out/output.csv")

# Select a random row
random_row = df.sample()

# Pretty print the content column
pprint.pprint(random_row['content'].values, indent=4)



import pandas as pd

def read_database(filename):
    df = pd.read_csv(filename)
    return df

def read_all_databases(namelist):
    dflist = []
    for filename in namelist:
        dflist.append(read_database(filename))
    stacked = pd.concat(dflist, axis=0)
    return stacked
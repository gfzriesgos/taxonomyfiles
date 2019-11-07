#!/usr/bin/env python3

import sys
import geopandas
import json
import pandas

def main():
    filename = sys.argv[1]

    data = geopandas.read_file(filename)

    s = 0

    for _, row in data.iterrows():
        expo = pandas.DataFrame(json.loads(row['expo']))
        s += expo['Buildings'].sum()

    print(s)

if __name__ == '__main__':
    main()

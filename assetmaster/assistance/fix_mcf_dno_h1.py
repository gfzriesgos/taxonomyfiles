#!/usr/bin/env python3

"""
The aim of this script is to fix
the taxonomy MCF-DNO-H1 to MCF-DNO-H1-3
"""

import json
import os

import geopandas
import pandas

def has_no_tax(data, base_tax):
    for _, row in data.iterrows():
        expo = pandas.DataFrame(json.loads(row['expo']))

        has_base_tax = (expo['Taxonomy'] == base_tax).any()
        if has_base_tax:
            return False

    return True


def main():
    base_file = 'Lima-Callao_SARA_Exposure_V3.gpkg'
    target_file = 'Lima-Callao_SARA_Exposure_V4.gpkg'

    base_tax = 'MCF-DNO-H1'
    target_tax = 'MCF-DNO-H1-3'

    data = geopandas.read_file(base_file)

    resulting_list = []

    for _, row in data.iterrows():
        expo = pandas.DataFrame(json.loads(row['expo']))

        # does we have the target_tax also all the time?
        target_expo = expo[expo['Taxonomy'] == target_tax]

        assert len(target_expo) == 1

        source_expo = expo[expo['Taxonomy'] == base_tax]

        assert len(source_expo) in [0, 1]

        if len(source_expo) == 1:
            buildings = source_expo.iloc[0]['Buildings']
            # change in place
            expo['Buildings'][expo['Taxonomy'] == target_tax] += buildings

        expo = expo[expo['Taxonomy'] != base_tax]

        assert not any(expo['Taxonomy'] == base_tax)

        row['expo'] = expo.to_json()

        resulting_list.append(row)

    data = pandas.DataFrame(resulting_list)
    data = geopandas.GeoDataFrame(data, geometry=data.geometry)

    assert has_no_tax(data, base_tax)

    data.to_file(target_file, 'GPKG')



if __name__ == '__main__':
    main()

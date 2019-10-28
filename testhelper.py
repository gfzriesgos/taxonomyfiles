#!/usr/bin/env python3

import collections
import glob
import json
import os
import pandas as pd
import geopandas as gpd

def nearly1(x):
    return 0.99 < x < 1.01

SourceTargetCombination = collections.namedtuple(
    'SourceTargetCombination',
    'source_schema target_schema'
)

def read_json(filename):
    with open(filename, 'rt') as input_file:
        data = json.load(input_file)
    return data

class FragilityData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_json_files(cls):
        fragility_models_by_schema = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        modelprop_dir = os.path.join(current_dir, 'modelprop')
        json_files = glob.glob(os.path.join(modelprop_dir, '*.json'))

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                schema = data['meta']['id']

                fragility_models_by_schema[schema] = data
        return cls(fragility_models_by_schema)

    def get_schemas(self):
        return self.data.keys()

    def get_taxonomies_in_meta(self, schema):
        return self.data[schema]['meta']['taxonomies']

    def get_taxonomies_in_data(self, schema):
        inner_data = self.data[schema]['data']
        results = []
        for dataset in inner_data:
            tax = dataset['taxonomy']
            results.append(tax)
        return results

    def get_imt_and_imu_values(self, schema):
        inner_data = self.data[schema]['data']

        results = collections.defaultdict(set)

        for dataset in inner_data:
            imt = dataset['imt']
            imu = dataset['imu']

            results[imt].add(imu)

        return results

    def get_damage_states_in_meta_with_D(self, schema):
        """
        Returns the damage states with D, so D0, D1, D2, ...
        """
        return self.data[schema]['meta']['limit_states']


class ExposureData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_json_files(cls):
        taxonomies_by_schema = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        assetmaster_dir = os.path.join(current_dir, 'assetmaster')
        json_files = glob.glob(os.path.join(assetmaster_dir, '*.json'))

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                schema = data['id']

                taxonomies = data['taxonomies']

                taxonomies_by_schema[schema] = taxonomies
        return cls(taxonomies_by_schema)

    def get_schemas(self):
        return self.data.keys()

    def get_taxonomies(self, schema):
        return self.data[schema]

class ExposureGeoData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_gpgk_files(cls):
        taxonomies_by_schema = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        assetmaster_dir = os.path.join(current_dir, 'assetmaster')
        gpkg_files = glob.glob(os.path.join(assetmaster_dir, '*.gpkg'))

        for gpkg_file in gpkg_files:
            taxonomies = set()
            geo_data = gpd.read_file(gpkg_file)
            for expo_str in geo_data['expo']:
                expo = pd.DataFrame(json.loads(expo_str))
                for tax in expo['Taxonomy']:
                    taxonomies.add(tax)

            base_filename = os.path.basename(gpkg_file)
            schema = base_filename.replace('_data.gpkg', '')
            taxonomies_by_schema[schema] = taxonomies

        return cls(taxonomies_by_schema)

    def get_schemas(self):
        return self.data.keys()

    def get_taxonomies(self, schema):
        return self.data[schema]


class SchemaMappingData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_json_files(cls):
        datasets = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        schemamapping_dir = os.path.join(current_dir, 'schemamapping')
        json_files = glob.glob(os.path.join(schemamapping_dir, '*', '*.json'))

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                datasets.append(data)

        return cls(datasets)

    def get_source_schemas(self):
        result = set()

        for dataset in self.data:
            source_schema = dataset['source_schema']
            result.add(source_schema)

        return result

    def get_source_taxonomies(self, schema):
        result = set()

        for dataset in self.data:
            source_schema = dataset['source_schema']
            if schema == source_schema:
                source_taxonomy = dataset['source_taxonomy']
                result.add(source_taxonomy)
        return result

    def get_target_schemas(self):
        result = set()
        for dataset in self.data:
            target_schema = dataset['target_schema']
            result.add(target_schema)
        return result

    def get_target_taxonomies(self, schema):
        result = set()
        for dataset in self.data:
            target_schema = dataset['target_schema']
            if schema == target_schema:
                target_taxonomy = dataset['target_taxonomy']
                result.add(target_taxonomy)
        return result

    def get_source_damage_states_from_conv_matrix_with_d(self, schema, taxonomy):
        result = set()
        for dataset in self.data:
            source_schema = dataset['source_schema']
            source_taxonomy = dataset['source_taxonomy']
            if source_schema == schema and source_taxonomy == taxonomy:
                conv_matrix = dataset['conv_matrix']
                result = conv_matrix.keys()

        result = ['D' + ds for ds in result]
        return result

    def get_source_damage_states_from_data_with_d(self, schema, taxonomy):
        result = set()
        for dataset in self.data:
            source_schema = dataset['source_schema']
            source_taxonomy = dataset['source_taxonomy']
            if source_schema == schema and source_taxonomy == taxonomy:
                result = result['source_damage_states']

        result = ['D' + ds for ds in result]
        return result

    def get_target_damage_states_from_conv_matrix_with_d(self, schema, taxonomy):
        result = set()
        for dataset in self.data:
            target_schema = dataset['target_schema']
            target_taxonomy = dataset['target_taxonomy']
            if target_schema == schema and target_taxonomy == taxonomy:
                conv_matrix = dataset['conv_matrix']
                for source_damage_state in conv_matrix.keys():
                    for target_damage_state in source_damage_state.keys():
                        result.add(target_damage_state)
        result = ['D' + ds for ds in result]
        return result

    def get_target_damage_states_from_data_with_d(self, schema, taxonomy):
        result = set()
        for dataset in self.data:
            target_schema = dataset['target_schema']
            target_taxonomy = dataset['target_taxonomy']
            if target_schema == schema and target_taxonomy == taxonomy:
                result = dataset['target_damage_states']
        result = ['D' + ds for ds in result]
        return result

class ReplacementcostData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_json_files(cls):
        replacement_costs_by_schema = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        replacementcost_dir = os.path.join(current_dir, 'replacement_costs')
        json_files = glob.glob(os.path.join(replacementcost_dir, '*.json'))

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)

                schema = data['meta']['id']

                replacement_costs_by_schema[schema] = data

        return cls(replacement_costs_by_schema)

    def get_schemas(self):
        return self.data.keys()

    def get_taxonomies(self, schema):
        result = set()
        for dataset in self.data[schema]['data']:
            tax = dataset['taxonomy']
            result.add(tax)
        return result

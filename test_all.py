#!/usr/bin/env python3

'''
The idea of this test case is to
handle all the schemas and taxonomies
consistent, so that they can run on deus
in later stages.

Just run the test with python3 test_all.py
'''

import collections
import glob
import json
import os
import unittest

import numpy as np
import geopandas as gpd
import pandas as pd

from testhelper import *

# This should be the names for the schemas
NAME_SARA = 'SARA_v1.0'
NAME_HAZUS = 'HAZUS_v1.0'
NAME_SUPPASRI = 'SUPPASRI2013_v2.0' 
NAME_MAVROULI = 'Mavrouli_et_al_2014'
NAME_TORRES = 'Torres_Corredor_et_al_2017'

class FileLoaderMixin():
    def load_exposure_json_file_sara(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_json_file = os.path.join(
            current_dir,
            'assetmaster', 
            'SARA_v1.0_asset_master.json'
        )

        json_exposure = read_json(exposure_json_file)
        return json_exposure

    def load_exposure_json_file_mavrouli(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_json_file = os.path.join(
            current_dir,
            'assetmaster',
            'Lahars_Mavrouli_et_al_2014_asset_master.json', 
        )

        json_exposure = read_json(exposure_json_file)
        return json_exposure
    def load_exposure_json_file_sara(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_json_file = os.path.join(
            current_dir,
            'assetmaster', 
            'SARA_v1.0_asset_master.json'
        )

        json_exposure = read_json(exposure_json_file)
        return json_exposure

    def load_exposure_gpkg_file_sara(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_gpkg_file = os.path.join(
            current_dir,
            'assetmaster',
            'SARA_v1.0_data.gpkg',
        )

        return gpd.read_file(exposure_gpkg_file)

    def load_exposure_gpkg_file_sara_peru(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_gpkg_file = os.path.join(
            current_dir,
            'assetmaster', 
            'Lima-Callao_SARA_Exposure_V4.gpkg',
        )

        return gpd.read_file(exposure_gpkg_file)

    def load_exposure_gpkg_file_mavrouli(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_gpkg_file = os.path.join(
            current_dir,
            'assetmaster',
            'Latacunga_Lahar_Mavrouli_et_al_2014_v1.0_data.gpkg', 
        )

        return gpd.read_file(exposure_gpkg_file)

    def load_exposure_gpkg_file_torres(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_gpkg_file = os.path.join(
            current_dir,
            'assetmaster',
            'Latacunga_Ash_Fall_Torres_Corredor_2017_v1.0_data.gpkg',
        )

        return gpd.read_file(exposure_gpkg_file)

    def load_fragility_sara(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fragility_json_file = os.path.join(
            current_dir,
            'modelprop',
            'SARA_v1.0_struct.json', 
        )

        return read_json(fragility_json_file)

    def load_fragility_mavrouli(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fragility_json_file = os.path.join(
            current_dir,
            'modelprop',
            'Mavrouli_et_al_2014_struct.json',
        )

        return read_json(fragility_json_file)

    def load_fragility_torres(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fragility_json_file = os.path.join(
            current_dir,
            'modelprop',
            'Ash-Fall_Torres_Corredor_et_al_2017-Fragility.json',
        )

        return read_json(fragility_json_file)

    def load_fragility_hazus(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fragility_json_file = os.path.join(
            current_dir,
            'modelprop',
            'HAZUS_v1.0_struct.json',
        )

        return read_json(fragility_json_file)

    def load_replacement_costs_sara(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        replacement_costs_json_file = os.path.join(
            current_dir,
            'replacement_costs', 
            'Replacement_cost_SARA_Seismic_v1.json',
        )

        return read_json(replacement_costs_json_file)

    def load_replacement_costs_suppasri(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        replacement_costs_json_file = os.path.join(
            current_dir,
            'replacement_costs',
            'Replacement_cost_SUPPASRI_Tsunami_v2.json', 
        )

        return read_json(replacement_costs_json_file)

    def load_replacement_costs_mavrouli(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        replacement_costs_json_file = os.path.join(
            current_dir,
            'replacement_costs',
            'Replacement_cost_MAVROULI_Lahar_v1.json',
        )

        return read_json(replacement_costs_json_file)

    def load_tax_schema_mapping_sara_to_suppasri(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tax_schema_mapping_json_file = os.path.join(
            current_dir,
            'tax_schemamappings', 
            'Conversion_Matrix_SARA_Suppasri_completed.json', 
        )

        return read_json(tax_schema_mapping_json_file)

    def load_fragility_suppasri(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fragility_json_file = os.path.join(
            current_dir,
            'modelprop', 
            'SUPPASRI2013_v2.0_struct.json',
        )

        return read_json(fragility_json_file)

    def load_ds_schema_mappings_sara_to_suppasri(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ds_schema_mapping_json_files = glob.glob(os.path.join(
            current_dir,
            'ds_schemamappings',
            '*', 
            '*.json', 
        ))

        ds_schema_mappings = [read_json(x) for x in ds_schema_mapping_json_files]

        result = [
            mapping for mapping in ds_schema_mappings
            if mapping['source_schema'] == NAME_SARA
            and mapping['target_schema'] == NAME_SUPPASRI
        ]

        return result




class DataGetterMixin():
    def get_taxonomies_from_exposure_gpkg(self, exposure_gpkg_data):
        gpkg_taxonomies = set()

        for _, row in exposure_gpkg_data.iterrows():
            expo = pd.DataFrame(json.loads(row['expo']))

            for tax in expo['Taxonomy'].unique():
                gpkg_taxonomies.add(tax)

        return gpkg_taxonomies

    def get_taxonomies_from_exposure_json(self, exposure_json_data):
        return exposure_json_data['taxonomies']

    def get_schema_from_exposure_json(self, exposure_json_data):
        return exposure_json_data['id']

    def get_schema_from_fragility(self, fragility_data):
        return fragility_data['meta']['id']

    def get_taxonomies_from_fragility(self, fragility_data):
        inner_data = fragility_data['data']

        taxonomies = set()

        for dataset in inner_data:
            taxonomy = dataset['taxonomy']
            taxonomies.add(taxonomy)

        return taxonomies

    def get_limit_states_from_fragility(self, fragility_data):
        return fragility_data['meta']['limit_states']

    def get_imts_from_fragility(self, fragility_data):
        imts = set()

        for dataset in fragility_data['data']:
            imt = dataset['imt']
            imts.add(imt)

        return imts

    def get_imus_from_fragility(self, fragility_data):
        imus = set()

        for dataset in fragility_data['data']:
            imu = dataset['imu']
            imus.add(imu)

        return imus

    def get_taxonomies_from_replacement_costs(self, replacement_cost_data):
        taxonomies = set()

        for dataset in replacement_cost_data['data']:
            taxonomy = dataset['taxonomy']
            taxonomies.add(taxonomy)

        return taxonomies

    def get_schema_from_replacement_costs(self, replacement_cost_data):
        return replacement_cost_data['meta']['id']

    def get_damage_states_from_fragility_without_D_prefix_as_int_with_0(self, fragility_data):
        limit_states = self.get_limit_states_from_fragility(fragility_data)

        without_d = [x[1:] for x in limit_states]
        as_int = [int(x) for x in without_d]

        with_0 = [0] + as_int

        return with_0

    def get_source_taxonomies_from_tax_schema_mapping(self, tax_schema_mapping_data):
        conv_matrix = tax_schema_mapping_data['conv_matrix']
        return conv_matrix.keys()

    def get_target_taxonomies_from_tax_schema_mapping(self, tax_schema_mapping_data):
        target_taxonomies = set()
        
        conv_matrix = tax_schema_mapping_data['conv_matrix']

        for source_taxonomy in conv_matrix.keys():
            for target_taxonomy in conv_matrix[source_taxonomy].keys():
                target_taxonomies.add(target_taxonomy)

        return target_taxonomies



class TaxonomyAssertionMixin():

    def assertExpsoureJsonSchemaMatches(self, exposure_json_data, schema):
        json_schema = self.get_schema_from_exposure_json(exposure_json_data)
        self.assertEqual(json_schema, schema)

    def assertFragilitySchemaMatches(self, fragility_data, schema):
        fragility_schema = self.get_schema_from_fragility(fragility_data)
        self.assertEqual(fragility_schema, schema)

    def assertTaxonomiesFromExposureJsonAndGpkgMatches(self, exposure_json_data, exposure_gpkg_data):
        json_taxonomies = self.get_taxonomies_from_exposure_json(exposure_json_data)
        gpkg_taxonomies = self.get_taxonomies_from_exposure_gpkg(exposure_gpkg_data)

        # we can't assume that all the taxonomies from the json file
        # are included but, we can assume that it must be the other way around
        self.assertAllTaxonomiesFromFirstAreInSecond(gpkg_taxonomies, json_taxonomies)

    def assertAllTaxonomiesFromFirstAreInSecond(self, first, second):
        missing = []
        for taxonomy in first:
            if taxonomy not in second:
                missing.append(taxonomy)

        self.assertEqual([], missing)

    def assertTaxonomiesFromExposureGpkgAndFragilitiesMatches(self, gpkg_data, fragility_data):
        gpkg_taxonomies = self.get_taxonomies_from_exposure_gpkg(gpkg_data)
        fragility_taxonomies = self.get_taxonomies_from_fragility(fragility_data)

        self.assertAllTaxonomiesFromFirstAreInSecond(gpkg_taxonomies, fragility_taxonomies)

    def assertTaxonomiesFromExposureJsonAndFragilitiesMatches(self, exposure_json_data, fragility_data):
        exposure_json_taxonomies = self.get_taxonomies_from_exposure_json(exposure_json_data)
        fragility_taxonomies = self.get_taxonomies_from_fragility(fragility_data)

        self.assertAllTaxonomiesFromFirstAreInSecond(exposure_json_taxonomies, fragility_taxonomies)

    def assertFragilityDamageStatesAreAllCovered(self, fragility_data):
        limit_states = self.get_limit_states_from_fragility(fragility_data)

        for dataset in fragility_data['data']:
            # we want to test every taxonomy on its own
            covered_damage_states = set()

            for possible_parameter in dataset.keys():
                if possible_parameter.startswith('D') and possible_parameter.endswith('_mean'):
                    damage_state = possible_parameter.replace('_mean', '')
                    covered_damage_states.add(damage_state)

            self.assertAllDamageStatesFromFirstAreInSecond(limit_states, covered_damage_states)

    def assertAllDamageStatesFromFirstAreInSecond(self, first, second):
        for ds in first:
            self.assertIn(ds, second)

    def assertFragilityImtsAreCoveredBySupportedImts(self, fragility_data, supported_imts):
        fragility_imts = self.get_imts_from_fragility(fragility_data)

        self.assertAllImtsFromFirstAreInSecond(fragility_imts, supported_imts)

    def assertAllImtsFromFirstAreInSecond(self, first, second):
        for imt in first:
            self.assertIn(imt, second)

    def assertFragilityImusAreCoveredBySupportedImus(self, fragility_data, supported_imus):
        fragility_imus = self.get_imus_from_fragility(fragility_data)
        self.assertAllImusFromFirstAreInSecond(fragility_imus, supported_imus)

    def assertAllImusFromFirstAreInSecond(self, first, second):
        for imu in first:
            self.assertIn(imu, second)

    def assertTaxonomiesFromExposureGpkgAndReplacementCostsMatches(self, gpkg_data, replacement_cost_data):
        gpkg_taxonomies = self.get_taxonomies_from_exposure_gpkg(gpkg_data)
        repl_cost_taxonomies = self.get_taxonomies_from_replacement_costs(replacement_cost_data)

        self.assertAllTaxonomiesFromFirstAreInSecond(gpkg_taxonomies, repl_cost_taxonomies)

    def assertTaxonomiesFromFragilityAndReplacementCostsMatches(self, fragility_data, replacement_cost_data):
        fragility_taxonimies = self.get_taxonomies_from_fragility(fragility_data)
        repl_cost_taxonomies = self.get_taxonomies_from_replacement_costs(replacement_cost_data)

        self.assertAllTaxonomiesFromFirstAreInSecond(fragility_taxonimies, repl_cost_taxonomies)

    def assertReplacementCostSchemaMatches(self, replacement_cost_data, schema):
        repl_cost_schema = self.get_schema_from_replacement_costs(replacement_cost_data)

        self.assertEqual(repl_cost_schema, schema)


    def assertFragilityDamageStatesAreAllCoveredByReplacementCosts(self, fragility_data, replacement_cost_data):
        damage_states = sorted(self.get_damage_states_from_fragility_without_D_prefix_as_int_with_0(fragility_data))
        all_but_highest = damage_states[:-1]

        for dataset in replacement_cost_data['data']:
            loss_matrix = dataset['loss_matrix']

            for ds in all_but_highest:
                # it is a string key
                self.assertIn(str(ds), loss_matrix.keys())

                all_higher = [x for x in damage_states if x > ds]

                for higher_ds in all_higher:
                    self.assertIn(str(higher_ds), loss_matrix[str(ds)].keys())

    def assertTaxonomiesFromExposureGpkgAndSourceTaxonomiesForTaxMappingMatches(self, gpkg_data, tax_schema_mapping_data):
        exposure_taxonomies = self.get_taxonomies_from_exposure_gpkg(gpkg_data)
        source_taxonomies = self.get_source_taxonomies_from_tax_schema_mapping(tax_schema_mapping_data)

        self.assertAllTaxonomiesFromFirstAreInSecond(exposure_taxonomies, source_taxonomies)

    def assertTaxonomySchemaMappingsSumTo1(self, tax_schema_mapping_data):
        conv_matrix = tax_schema_mapping_data['conv_matrix']

        for source_taxonomy in conv_matrix.keys():
            sum_so_far = 0.0

            for target_taxonomy in conv_matrix[source_taxonomy].keys():
                val = conv_matrix[source_taxonomy][target_taxonomy]
                sum_so_far += val

            self.assertInRange(0.99, sum_so_far, 1.01)

    def assertInRange(self, lower, x, upper):
        self.assertLess(lower, x)
        self.assertLess(x, upper)

    def assertTaxonomyTargetSchemaMappingCoveredByFragilityModel(self, tax_schema_mapping_data, fragility_data):
        target_taxonimies = self.get_target_taxonomies_from_tax_schema_mapping(tax_schema_mapping_data)
        fragility_taxonomies = self.get_taxonomies_from_fragility(fragility_data)

        self.assertAllTaxonomiesFromFirstAreInSecond(target_taxonimies, fragility_taxonomies)

    def assertAllFragilityDamageStatesCoveredBySchemaSourceDamageStates(self, fragility_data, ds_schema_mappings):
        # important to have it here as strings
        damage_states = [str(x) for x in self.get_damage_states_from_fragility_without_D_prefix_as_int_with_0(fragility_data)]

        for mapping_dataset in ds_schema_mappings:
            conv_matrix = mapping_dataset['conv_matrix']
            for target_damage_state in conv_matrix.keys():
                source_damage_states = conv_matrix[target_damage_state].keys()

                self.assertAllDamageStatesFromFirstAreInSecond(damage_states, source_damage_states)

    def assertAllTargetDamageStatesCoveredByFragilityDamageStates(self, ds_schema_mappings, fragility_data):
        damage_states = [str(x) for x in self.get_damage_states_from_fragility_without_D_prefix_as_int_with_0(fragility_data)]

        for mapping_dataset in ds_schema_mappings:
            conv_matrix = mapping_dataset['conv_matrix']
            self.assertAllDamageStatesFromFirstAreInSecond(conv_matrix.keys(), damage_states)


    def assertDamageStateMappingsForAllSourceTargetCombinationsByTaxMapping(self, ds_schema_mappings, tax_schema_mapping_data):
        MissingSourceTargetDsMapping = collections.namedtuple('MissingSourceTargetDsMapping', 'source_taxonomy target_taxonomy')
        missing_mappings = set()

        for source_taxonomy in tax_schema_mapping_data['conv_matrix'].keys():
            for target_taxonomy in tax_schema_mapping_data['conv_matrix'][source_taxonomy]:

                #happens = tax_schema_mapping_data['conv_matrix'][source_taxonomy][target_taxonomy] > 0.0

                possible_ds_mappings = [
                    x for x in ds_schema_mappings
                    if x['source_taxonomy'] == source_taxonomy
                    and x['target_taxonomy'] == target_taxonomy
                ]

                if not possible_ds_mappings:
                    missing_mappings.add(MissingSourceTargetDsMapping(source_taxonomy=source_taxonomy, target_taxonomy=target_taxonomy))

        self.assertEqual(set(), missing_mappings)

    def assertDamageStateMappingsSourceSchemaMatches(self, ds_schema_mappings, schema):
        for dataset in ds_schema_mappings:
            source_schema = dataset['source_schema']
            self.assertEqual(source_schema, schema)

    def assertDamageStateMappingsTargetSchemaMatches(self, ds_schema_mappings, schema):
        for dataset in ds_schema_mappings:
            target_schema = dataset['target_schema']
            self.assertEqual(target_schema, schema)


    def assertGpkgExposureStructureMatches(self, gpkg_data):
        expected_top_level_columns = ['gid', 'name', 'expo', 'geometry']
        actual_columns = gpkg_data.columns

        for column in expected_top_level_columns:
            self.assertIn(column, actual_columns)

        for column in actual_columns:
            if column != 'id':
                self.assertIn(column, expected_top_level_columns)

        expected_expo_columns = ['id', 'Region', 'Taxonomy', 'Dwellings', 'Buildings', 'Repl-cost-USD-bdg', 'Population', 'name', 'Damage']

        for _, row in gpkg_data.iterrows():
            self.assertEqual(type(row['expo']), str)

            expo = pd.DataFrame(json.loads(row['expo']))

            actual_expo_columns = expo.columns

            self.assertEqual(set(expected_expo_columns), set(actual_expo_columns))
            self.assertIn(expo.dtypes['Buildings'], [np.dtype('float64'), np.dtype('int64')])

    def assertFragilityStructureMatches(self, fragility_data):
        # we need to have the schema in the id field
        self.assertIn('meta', fragility_data.keys())
        self.assertIn('id', fragility_data['meta'].keys())

        self.assertIsNotNone(fragility_data['meta']['id'])

        # then we need to have the shape value
        self.assertIn('shape', fragility_data['meta'].keys())
        self.assertIsNotNone(fragility_data['meta']['shape'])

        # then we need to have the data
        self.assertIn('data', fragility_data.keys())
        # this must be an array
        # with contents
        self.assertLess(0, len(fragility_data['data']))

        for dataset in fragility_data['data']:
            self.assertIn('taxonomy', dataset.keys())
            self.assertIsNotNone(dataset['taxonomy'])
            self.assertIn('imt', dataset.keys())
            self.assertIsNotNone(dataset['imt'])
            self.assertIn('imu', dataset.keys())
            self.assertIsNotNone(dataset['imu'])



    def assertFragilityShapesAreCoveredBySupportedShapes(self, fragility_data, supported_schapes):
        shape = fragility_data['meta']['shape']
        self.assertIn(shape, supported_schapes)

class TestSaraPeru(unittest.TestCase, TaxonomyAssertionMixin, FileLoaderMixin, DataGetterMixin):
    """
    This is the test case for the exposure model for peru (also sara).
    """

    def test_all_sara_taxonomies_in_exposure_model_are_defined_in_json(self):
        """
        This test ensures that all of the taxonomies used in the
        exposure model for peru are also in the json exposure file
        for for sara (defining the taxonomies).
        """
        json_exposure = self.load_exposure_json_file_sara()
        gpkg_exposure = self.load_exposure_gpkg_file_sara_peru()
        self.assertTaxonomiesFromExposureJsonAndGpkgMatches(json_exposure, gpkg_exposure)


    def test_gpkg_has_expected_structure(self):
        """
        This is the test to ensure that the gpkg file has the structure to
        work with it in deus.
        """

        gpkg_exposure_peru = self.load_exposure_gpkg_file_sara_peru()
        self.assertGpkgExposureStructureMatches(gpkg_exposure_peru)

    def test_all_sara_taxonomies_in_exposure_model_have_fragility(self):
        """
        This is the next step.
        Because we use deus, we want to have fragility functons
        for all of the taxonomies, so that we can be sure that
        we don't have a taxonomy in there that is not covered
        in the fragility taxonomies.
        """
        gpkg_exposure = self.load_exposure_gpkg_file_sara_peru()
        fragility = self.load_fragility_sara()
        
        # first make sure that we have the right schema name in there
        self.assertFragilitySchemaMatches(fragility, NAME_SARA)

        # then we want to make sure that we cover all of the
        # taxonomies that we have in the exposure model
        self.assertTaxonomiesFromExposureGpkgAndFragilitiesMatches(gpkg_exposure, fragility)

    def test_all_sara_taxonomies_in_exposure_model_have_replacement_costs(self):
        """
        When we want to compute the loss we need the replacement costs
        and we need that it covers all of the taxonomies from the exposure model.
        """

        gpkg_exposure = self.load_exposure_gpkg_file_sara_peru()
        replacement_costs = self.load_replacement_costs_sara()

        self.assertTaxonomiesFromExposureGpkgAndReplacementCostsMatches(gpkg_exposure, replacement_costs)
        self.assertReplacementCostSchemaMatches(replacement_costs, NAME_SARA)

class TestSara(unittest.TestCase, TaxonomyAssertionMixin, FileLoaderMixin, DataGetterMixin):
    """
    This test case is for testing the chain for the sara model.
    """

    def test_all_sara_taxonomies_in_exposure_model_are_defined_in_json(self):
        """
        Tests that all the taxonomies that are in
        the sara exposure model (the gpkg) are also defined
        in the sara json file.

        This is the first step to make sure, so that we can be very
        sure that our gpkg model is correct.
        """
        json_exposure = self.load_exposure_json_file_sara()
        self.assertExpsoureJsonSchemaMatches(json_exposure, NAME_SARA)

        gpkg_exposure = self.load_exposure_gpkg_file_sara()
        self.assertTaxonomiesFromExposureJsonAndGpkgMatches(json_exposure, gpkg_exposure)

    def test_gpkg_has_expected_structure(self):
        """
        This is the test to check that the gpkg file has the structure
        to work with it in deus.
        """

        gpkg_exposure = self.load_exposure_gpkg_file_sara()
        self.assertGpkgExposureStructureMatches(gpkg_exposure)

    def test_fragility_model_has_expected_structure(self):
        """
        This is the test to check that the structure of the fragility
        file is as expected.
        """
        fragility = self.load_fragility_sara()
        self.assertFragilityStructureMatches(fragility)

    def test_all_sara_taxonomies_in_exposure_model_have_fragility(self):
        """
        This is the next step.
        Because we use deus, we want to have fragility functons
        for all of the taxonomies, so that we can be sure that
        we don't have a taxonomy in there that is not covered
        in the fragility taxonomies.
        """
        gpkg_exposure = self.load_exposure_gpkg_file_sara()
        fragility = self.load_fragility_sara()
        
        # first make sure that we have the right schema name in there
        self.assertFragilitySchemaMatches(fragility, NAME_SARA)

        # then we want to make sure that we cover all of the
        # taxonomies that we have in the exposure model
        self.assertTaxonomiesFromExposureGpkgAndFragilitiesMatches(gpkg_exposure, fragility)

    def test_all_sara_taxonomies_in_exposure_json_have_fragility(self):
        """
        This test also tries if all the taxonomies from the expose
        json are included in the fragility model too.
        """
        json_exposure = self.load_exposure_json_file_sara()
        fragility = self.load_fragility_sara()

        self.assertTaxonomiesFromExposureJsonAndFragilitiesMatches(json_exposure, fragility)

    def test_fragility_sara_covers_all_damage_states(self):
        """
        We want also to be sure that our fragility functions
        cover all the damage states, that we need.
        """

        fragility = self.load_fragility_sara()
        self.assertFragilityDamageStatesAreAllCovered(fragility)

    def test_fragility_sara_imts(self):
        """
        We only support some imts.
        """

        fragility = self.load_fragility_sara()
        supported_imts = ['PGA', 'SA(1.0)', 'SA(0.3)']

        self.assertFragilityImtsAreCoveredBySupportedImts(fragility, supported_imts)

    def test_fragility_sara_imus(self):
        """
        We also only support some limited imu values.
        """
        fragility = self.load_fragility_sara()
        supported_imus = ['g']

        self.assertFragilityImusAreCoveredBySupportedImus(fragility, supported_imus)

    def test_fragility_shape(self):
        """
        We must make sure that a function is used that we can deal with.
        """
        fragility = self.load_fragility_sara()
        supported_shapes = ['logncdf']

        self.assertFragilityShapesAreCoveredBySupportedShapes(fragility, supported_shapes)

    def test_all_sara_taxonomies_in_exposure_model_have_replacement_costs(self):
        """
        When we want to compute the loss we need the replacement costs
        and we need that it covers all of the taxonomies from the exposure model.
        """

        gpkg_exposure = self.load_exposure_gpkg_file_sara()
        replacement_costs = self.load_replacement_costs_sara()

        self.assertTaxonomiesFromExposureGpkgAndReplacementCostsMatches(gpkg_exposure, replacement_costs)
        self.assertReplacementCostSchemaMatches(replacement_costs, NAME_SARA)

    def test_replacement_costs_sara_cover_all_damage_states(self):
        """
        Here we test that all of our damage states are covered in the
        replacement costs.
        """
        fragility = self.load_fragility_sara()
        replacement_costs = self.load_replacement_costs_sara()

        self.assertFragilityDamageStatesAreAllCoveredByReplacementCosts(fragility, replacement_costs)


class TestSaraToSuppasri(unittest.TestCase, TaxonomyAssertionMixin, FileLoaderMixin, DataGetterMixin):
    """
    This is the test case explicit for the chain from sara to suppasri.
    """

    def test_all_sara_taxonomies_are_covered_by_tax_schema_mapping(self):
        """
        This is the test that we can map all of the taxonomies from
        sara to suppasri (so all are covered).
        """

        gpkg_exposure_sara = self.load_exposure_gpkg_file_sara()
        tax_schema_mappings_sara_to_suppasri = self.load_tax_schema_mapping_sara_to_suppasri()

        self.assertTaxonomiesFromExposureGpkgAndSourceTaxonomiesForTaxMappingMatches(
            gpkg_exposure_sara,
            tax_schema_mappings_sara_to_suppasri,
        )

    def test_target_taxonomies_sum_to_1(self):
        """
        We must make sure that all we don't add or remove a
        building by the schema mapping for the taxonomy mapping.
        """
        tax_schema_mappings_sara_to_suppasri = self.load_tax_schema_mapping_sara_to_suppasri()
        self.assertTaxonomySchemaMappingsSumTo1(tax_schema_mappings_sara_to_suppasri)

    def test_target_taxonomies_are_in_fragility_model(self):
        """
        We also must make sure that we have fragility models for all of your target
        taxonomies.
        """
        tax_schema_mappings_sara_to_suppasri = self.load_tax_schema_mapping_sara_to_suppasri()
        fragility = self.load_fragility_suppasri()

        self.assertTaxonomyTargetSchemaMappingCoveredByFragilityModel(
            tax_schema_mappings_sara_to_suppasri, 
            fragility
        )

    def test_damage_state_specifics_cover_all_tax_mappings(self):
        """
        We must make sure that we can be sure, that all the damage state
        mappings (that are specific to the source and target taxonomies)
        are covered. As source we use the tax schema mapping.
        """
        tax_schema_mapping = self.load_tax_schema_mapping_sara_to_suppasri()
        ds_schema_mappings = self.load_ds_schema_mappings_sara_to_suppasri()

        self.assertDamageStateMappingsForAllSourceTargetCombinationsByTaxMapping(
            ds_schema_mappings,
            tax_schema_mapping
        )

    def test_all_source_damage_states_in_ds_schema_mappings(self):
        """
        This is the test to check if all of our source damage states
        are covered by the the damage state mapping.
        Since this they are specific for each taxonomy this is way harder
        than before.
        """

        fragility_sara = self.load_fragility_sara()
        ds_schema_mapingss_sara_to_suppasri = self.load_ds_schema_mappings_sara_to_suppasri()

        self.assertAllFragilityDamageStatesCoveredBySchemaSourceDamageStates(fragility_sara, ds_schema_mapingss_sara_to_suppasri)

    def test_all_target_damage_states_in_fragility_model(self):
        """
        We must make sure that all our target damage states have
        asociated fragility functions.
        """
        fragility_suppasri = self.load_fragility_suppasri()

        ds_schema_mapingss_sara_to_suppasri = self.load_ds_schema_mappings_sara_to_suppasri()

        self.assertAllTargetDamageStatesCoveredByFragilityDamageStates(ds_schema_mapingss_sara_to_suppasri, fragility_suppasri)

    def test_all_damage_state_mappings_use_right_schemas(self):
        """
        We must make sure that the source schema is right.
        """

        ds_schema_mapingss_sara_to_suppasri = self.load_ds_schema_mappings_sara_to_suppasri()

        self.assertDamageStateMappingsSourceSchemaMatches(ds_schema_mapingss_sara_to_suppasri, NAME_SARA)
        self.assertDamageStateMappingsTargetSchemaMatches(ds_schema_mapingss_sara_to_suppasri, NAME_SUPPASRI)

    def test_suppasri_fragility_schema(self):
        fragility = self.load_fragility_suppasri()

        self.assertFragilitySchemaMatches(fragility, NAME_SUPPASRI)

    def test_suppasri_fragility_imts(self):
        fragility = self.load_fragility_suppasri()

        supported_imts = ['MWH', 'ID']
        self.assertFragilityImtsAreCoveredBySupportedImts(fragility, supported_imts)

    def test_suppasri_fragility_imus(self):
        fragility = self.load_fragility_suppasri()

        supported_imus = ['m']
        self.assertFragilityImusAreCoveredBySupportedImus(fragility, supported_imus)

    def test_all_suppasri_taxonomies_in_fragility_model_have_replacement_costs(self):
        """
        When we map from sara to suppasri we have to make sure,
        that we also have the replacement costs for all the
        taxonomies in the fragility model.
        """
        fragility = self.load_fragility_suppasri()
        replacement_costs = self.load_replacement_costs_suppasri()

        self.assertReplacementCostSchemaMatches(replacement_costs, NAME_SUPPASRI)
        self.assertTaxonomiesFromFragilityAndReplacementCostsMatches(fragility, replacement_costs)

    def test_replacement_costs_suppasri_cover_all_damage_states(self):
        """
        Here we make sure that we cover all the damage states in the replacement
        costs.
        """

        fragility = self.load_fragility_suppasri()
        replacement_costs = self.load_replacement_costs_suppasri()

        self.assertFragilityDamageStatesAreAllCoveredByReplacementCosts(fragility, replacement_costs)


class TestTaxMappingFiles(unittest.TestCase):
    """
    This Test case is only for the taxonomy mapping files.
    """

    def setUp(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_files = glob.glob(os.path.join(current_dir, 'tax_schemamappings', '*.json'))

        self.json_data = []

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                self.json_data.append(data)


    def test_all_tax_mapping_files_sum_to_1(self):
        """
        The sum for a source taxonomies over all
        target taxonomies must be 1.
        """

        for dataset in self.json_data:
            conv_matrix = dataset['conv_matrix']

            for source_taxonomy in conv_matrix.keys():
                conv_row = conv_matrix[source_taxonomy]

                sum_over_row = sum(conv_row.values())

                self.assertTrue(nearly1(sum_over_row))

    def test_tax_mapping_supported(self):
        """
        Test that all the taxomomy mapping files have the supported
        schemas.
        """


        schema_mappings = set()
        for dataset in self.json_data:
            source_schema = dataset['source_schema']
            target_schema = dataset['target_schema']
            schema_mappings.add(
                SourceTargetCombination(
                    source_schema=source_schema,
                    target_schema=target_schema,
                )
            )

        # we need to support the mapping from
        # sara to supparsi
        self.assertIn(
            SourceTargetCombination(
                source_schema=NAME_SARA, 
                target_schema=NAME_SUPPASRI,
            ),
            schema_mappings
        )

    def test_covers_full_sara_to_suppasri_for_all_tax(self):
        """
        Reads the taxomomy from the exposure json file for sara
        and checks if the all the taxonomies of sara are
        supported.
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_file_sara = os.path.join(
            current_dir, 
            'assetmaster',
            'SARA_v1.0_asset_master.json',
        )

        exposure_sara = read_json(exposure_file_sara)

        sara_taxonomies = exposure_sara['taxonomies']

        covered_taxonomies = set()

        for dataset in self.json_data:
            source_schema = dataset['source_schema']
            target_schema = dataset['target_schema']

            if source_schema == NAME_SARA and target_schema == NAME_SUPPASRI:
                conv_matrix = dataset['conv_matrix']
                taxonomies = conv_matrix.keys()

                for tax in taxonomies:
                    covered_taxonomies.add(tax)

        not_covered_taxonomies = set(sara_taxonomies) - covered_taxonomies

        self.assertEqual(not_covered_taxonomies, set())



class TestDsMappingFiles(unittest.TestCase):
    """
    This test case is only for the damage state mapping
    files.
    """

    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_files = glob.glob(os.path.join(current_dir, 'ds_schemamappings', '*', '*.json'))

        self.json_data = []
        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                self.json_data.append(data)


    def test_all_ds_mapping_files_sum_to_1(self):
        """
        The sum for each damage state mapping between
        schemas must be 1.
        """

        for dataset in self.json_data:
            # because the ds files have the target damage states
            # as toplevel keys, we can use the pandas dataframe
            conv_matrix = pd.DataFrame(dataset['conv_matrix'])

            self.assertTrue(all(conv_matrix.apply(sum, axis=1).apply(nearly1)))

    def test_tax_mapping_supported(self):
        """
        Test that all the taxomomy mapping files have the supported
        schemas.
        """


        schema_mappings = set()
        for dataset in self.json_data:
            source_schema = dataset['source_schema']
            target_schema = dataset['target_schema']
            schema_mappings.add(
                SourceTargetCombination(
                    source_schema=source_schema,
                    target_schema=target_schema,
                )
            )

        # we need to support the mapping from
        # sara to supparsi
        self.assertIn(
            SourceTargetCombination(
                source_schema=NAME_SARA, 
                target_schema=NAME_SUPPASRI,
            ),
            schema_mappings
        )

    def test_covers_full_sara_to_suppasri_for_all_tax(self):
        """
        Reads the taxonomy from the exposure json file for sara
        and checks if all the taxonomies of sara are supported.
        """


        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_file_sara = os.path.join(
            current_dir, 
            'assetmaster',
            'SARA_v1.0_asset_master.json',
        )

        exposure_sara = read_json(exposure_file_sara)

        sara_taxonomies = exposure_sara['taxonomies']

        covered_taxonomies = set()

        for dataset in self.json_data:
            source_schema = dataset['source_schema']
            target_schema = dataset['target_schema']

            if source_schema == NAME_SARA and target_schema == NAME_SUPPASRI:
                source_taxonomy = dataset['source_taxonomy']

                covered_taxonomies.add(source_taxonomy)

        not_covered_taxonomies = set(sara_taxonomies) - covered_taxonomies

        self.assertEqual(not_covered_taxonomies, set())

    def test_covers_full_sara_to_suppasri_for_all_ds(self):
        """
        Test that we cover all source damage states
        in all the files for the sara to suppasri
        mappings.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fragility_file_sara = os.path.join(
            current_dir, 
            'modelprop',
            'SARA_v1.0_struct.json',
        )

        fragility_sara = read_json(fragility_file_sara)

        sara_ds = set(['0'] + [x[1:] for x in fragility_sara['meta']['limit_states']])

        fragility_file_suppasri = os.path.join(
            current_dir,
            'modelprop',
            'SUPPASRI2013_v2.0_struct.json'
        )

        fragility_suppasri = read_json(fragility_file_suppasri)

        suppasri_ds = set(['0'] + [x[1:] for x in fragility_suppasri['meta']['limit_states']])

        for dataset in self.json_data:
            source_taxonomy = dataset['source_taxonomy']
            target_taxonomy = dataset['target_taxonomy']

            if source_taxonomy == NAME_SARA and target_taxonomy == NAME_SUPPASRI:
                conv_matrix = pd.DataFrame(dataset['conv_matrix'])

                source_ds = set(conv_matrix.index)
                target_ds = set(conv_matrix.columns)

                # we need to support the full range of input damage states
                self.assertEqual(source_ds, sara_ds)

                for single_target_ds in target_ds:
                    # and the target damage states must be those
                    # that are defined
                    self.assertIn(single_target_ds, suppasri_ds)

class TestMavrouliEcuador(unittest.TestCase, TaxonomyAssertionMixin, FileLoaderMixin, DataGetterMixin):
    """
    This is the test case for the Mavrouli files (lahar).
    """

    def test_fragility_file_matches_expected_structure(self):
        fragility = self.load_fragility_mavrouli()
        self.assertFragilityStructureMatches(fragility)

    def test_all_mavrouli_taxonomies_in_exposure_model_are_defined_in_json(self):
        """
        Because we have a list of all the taxonomies that can
        be there in the exposure model for lahars, we want to test it.
        """
        json_exposure = self.load_exposure_json_file_mavrouli()
        gpkg_exposure = self.load_exposure_gpkg_file_mavrouli()
        self.assertTaxonomiesFromExposureJsonAndGpkgMatches(json_exposure, gpkg_exposure)

    def test_gpkg_has_expected_structure(self):
        """
        Test for the structure of the gkpg file.
        """
        gpkg_exposure_mavrouli = self.load_exposure_gpkg_file_mavrouli()
        self.assertGpkgExposureStructureMatches(gpkg_exposure_mavrouli)

    def test_all_mavrouli_taxonomies_in_exposure_model_have_replacement_costs(self):
        """
        Tests that we have replacement costs for all the mavrouli taxonomies.
        """
        gpkg_exposure = self.load_exposure_gpkg_file_mavrouli()
        replacement_costs = self.load_replacement_costs_mavrouli()
        self.assertTaxonomiesFromExposureGpkgAndReplacementCostsMatches(gpkg_exposure, replacement_costs)
        self.assertReplacementCostSchemaMatches(replacement_costs, NAME_MAVROULI)

class TestTorresEcuador(unittest.TestCase, TaxonomyAssertionMixin, FileLoaderMixin, DataGetterMixin):
    """
    This is the test case for the Torres files (ashfalls).
    """

    def test_fragility_file_matches_expected_structure(self):
        fragility = self.load_fragility_torres()
        self.assertFragilityStructureMatches(fragility)

    def test_gpkg_has_expected_structure(self):
        """
        Test for the structure of the gkpg file.
        """
        gpkg_exposure_torres = self.load_exposure_gpkg_file_torres()
        self.assertGpkgExposureStructureMatches(gpkg_exposure_torres)

class TestHazus(unittest.TestCase, TaxonomyAssertionMixin, FileLoaderMixin, DataGetterMixin):
    """
    This is the test case for the hazus files (also earthquakes.
    """

    def test_fragility_file_matches_expected_structure(self):
        fragility = self.load_fragility_hazus()
        self.assertFragilityStructureMatches(fragility)

if __name__ == '__main__':
    unittest.main()

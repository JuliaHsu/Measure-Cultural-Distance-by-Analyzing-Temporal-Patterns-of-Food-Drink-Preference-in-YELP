'''Convert the Yelp Dataset Challenge dataset from json format to csv.'''

import argparse
import collections
from collections.abc import MutableMapping
import csv
import simplejson as json
import glob

DATA = '../yelp_dataset/'
def read_and_write_file(json_file_path, csv_file_path, column_names):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    with open(csv_file_path, 'w+',newline='') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        # l = 1
        with open(json_file_path) as fin:
            for line in fin:
                # print("line: "+ str(l))
                line_contents = json.loads(line)
                csv_file.writerow(get_row(line_contents, column_names))
                # l+=1

def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""

    column_names = set()
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            
            column_names.update(
                    set(get_column_names(line_contents).keys())
                    )
    return column_names


def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.

    Example:

        line_contents = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }

        will return: ['a.b', 'a.c']

    These will be the column names for the eventual csv file.

    """
    column_names = []
    for k, v in line_contents.items():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, MutableMapping):
            column_names.extend(
                    get_column_names(v, column_name).items()
                    )
        else:
            column_names.append((column_name, v))
    return dict(column_names)

def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        # print(column_name)
        line_value = get_nested_value(
                        line_contents,
                        column_name,
                        )
        # if isinstance(line_value, str):
        #     row.append('{0}'.format(line_value))
        if line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row

def get_nested_value(d, key):
    """Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.
    d = content of each line
    Example:

        d = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        key = 'a.b'

        will return: 2
    
    
    """
    
    if '.' not in key:
        if d is None:
            return None
        elif key not in d:
            return None # end of traverse (no subkey in key)
        return d[key] # the key is in dictionary and the key does not contain nested key-value, return the value of (sub)key (no nested value)
    base_key, sub_key = key.split('.', 1) 
    if base_key not in d:
        return None
    sub_dict = d[base_key] # value of basekey (could be a dictionary of dictionary). e.g., attributes.ByAppointmentOnly --> sub_dict = d[attributes] (= {'GoodForKids': 'True', 'ByAppointmentOnly': 'True'}, if the subkey is not in sub_dict, return none, o.w. return the value of subkey)


    return get_nested_value(sub_dict, sub_key) # traverse to the sub_key. e.g, a.b -> b.c (or b, if there's no subkey in b) 

# convert every json file to csv file
json_files = glob.glob(DATA +"*.json")
for json_file in json_files:
    csv_file = json_file.replace('.json','.csv')  
    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)


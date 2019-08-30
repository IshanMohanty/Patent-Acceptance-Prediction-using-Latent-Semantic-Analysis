"""
@author: imohanty
"""
#dependencies
import json
from pandas.io.json import json_normalize

'''
json_to_df function:
Converts json file to dataframe format
@return df: dataframe format from json format
'''
def json_to_df(file_name):
    
    '''
    Read individual json files from the given json file. The given json file
    suffers from a multiroot problem. (Hence, direct extraction is not possible)
    Each json file is stored as string.
    '''
    json_files = []
    for json_file in open(file_name, 'r'):
        json_files.append(json.loads(json_file))
    
    '''
    convert the string format json file to an actual json file format
    '''    
    x = json.dumps(json_files[0])
    x = json.loads(x)
    
    #expand the columns/features of the json file and convert to dataframe
    df  = json_normalize(x['object'])
    
    #convert and append to dataframe
    for json_file in json_files[1:]:
        x = json.dumps(json_file)
        x = json.loads(x)
        x = json_normalize(x['object'])
        df = df.append([x],sort=False)
    
    '''
    Construct labels for patent present by searching 'Patentend Case' string.
    label as 1 if 'Patented Case' found in status else label 0 as Non-Patented
    Case.
    '''    
    df['Decision'] = 0
    df['Decision'] = df['status'].str.contains('Patented Case')
    df.loc[df['Decision'] == True, 'Decision'] = 1
    df.loc[df['Decision'] == False, 'Decision'] = 0
    
    return df
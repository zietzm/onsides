import numpy as np 
import pandas as pd
import requests
from tqdm import tqdm
import ast, re
from time import sleep
import argparse
import warnings
warnings.filterwarnings('ignore')
import os
import collections


def main():
    parser = argparse.ArgumentParser(description='let the code know where the data is held')
    parser.add_argument('--data_folder', required=True, help='Path to the data folder.')
    parser.add_argument('--external_data', required=True, help='Path to the where the external data is housed.')
    #parser.add_argument('--final_data', required=True, help='Path to the where the final output should be written.')
    parser.add_argument('--model', required=True, help='Path to the where the model is housed.')
    parser.add_argument('--commands', required=True, help='Path to the where the commands are housed.')

    args = parser.parse_args()
    data_folder = args.data_folder
    external_data_folder = args.external_data
    #final_folder = args.final_data
    model = args.model
    commands = args.commands

    #Step 0: check if the drug data has been downloaded. If not, return an error
    if not os.path.exists(data_folder+'drug_data.csv'):
        print('drug data not found. Please run download_files.py first.')
        return None
    
    #Step 1: parse the drug data
    print('we found the drug data! now, let\'s parse it.')
    parse_cmd = commands+' parse_files.py --data_folder '+data_folder
    print('running parse.py')
    os.system('python3 '+parse_cmd)
    print('finished parse.py')

    #Step 2: extract ades from tabular drug data
    print('we parsed the drug data! now, let\'s extract the ades.')
    extract_cmd = commands+' tabular_data_mapping.py --data_folder '+data_folder+' --external_data '+external_data_folder
    print('running extract_ades.py')
    os.system('python3 '+extract_cmd)
    print('finished extract_ades.py')

    #Step 3: extract ades from free text data
    print('we extracted the ades! now, let\'s extract the ades from the free text.')
    text_cmd = commands+' text_data_format.py --data_folder '+data_folder+' --external_data '+external_data_folder
    print('running text_data_format.py')
    os.system('python3 '+text_cmd)
    print('finished text_data_format.py')

    #Step 4: predict the ades from the free text data using the OnSIDES model
    print('we extracted the ades from the free text! now, let\'s predict the ades from the free text.')
    predict_cmd = commands+' text_data_predict.py --data_folder '+data_folder+' --external_data '+external_data_folder+' --model_path '+model
    print('running text_data_predict.py')
    os.system('python3 '+predict_cmd)
    print('finished text_data_predict.py')

    print('##############################################')
    print('finished all steps!')
    print('##############################################')
    print('now, let\'s build the OnSIDES UK data. run build_onsides_uk.py')

if __name__ == '__main__':
    main()
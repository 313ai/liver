import requests
import os
import pickle
from pprint import pprint
import ipdb


html_encoding = {
    "\"": "%22", 
    ":" : "%3A",
    "{" : "%7B",
    "}" : "%7D",
    "," : "%2C",
    "[" : "%5B",
    "]" : "%5D"
    }

def get_diagnoses(case_id='TCGA-DD-AADF'):
    # https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/#filtering-operators
    
    ## get request
    payload = "{\"op\":\"and\",\
               \"content\":[\
                    {\"op\":\"in\",\"content\":{\
                        \"field\":\"submitter_id\",\
                        \"value\":[\"%s\"]}\
                    }]\
               }" % case_id

    escaped_payload  = ("".join(html_encoding.get(c,c) for c in payload)).replace(" ","")
    url = 'https://api.gdc.cancer.gov/cases?filters='

    response = requests.get(url+escaped_payload)

    #get the case ID / ID so we can lookup diagnosis:
    uuid = response.json()['data']['hits'][0]['id']

    url_2 = 'https://api.gdc.cancer.gov/cases/%s?pretty=true&expand=diagnoses' % uuid
    response = requests.get(url_2).json()
    diagnoses_dict = response['data']['diagnoses'][0].copy()

    return diagnoses_dict


if __name__ == "__main__":
    # get_diagnoses()

    out_dict = {}

    root_dir = './data/'
    i = 0
    for dir_name,sub_dir_list, filelist in os.walk(root_dir):
        try:
            slide_filename = [x for x in filelist if ('.svs' in x and '.parcel' not in x)][0]
        except IndexError:
            slide_filename = []
        if len(slide_filename) > 0:
            case_id = slide_filename[:12]
            print(i,slide_filename)
            
            out_dict[slide_filename] = {
                'case_id':case_id,
                'dir_name':dir_name,
                'diagnoses':get_diagnoses(case_id)
            }
            i += 1
            
    with open('diagnoses_dict.pkl','wb') as f:
        pickle.dump(out_dict,f,pickle.HIGHEST_PROTOCOL)




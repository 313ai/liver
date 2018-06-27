import requests
import json
import pandas as pd

cases_endpt = 'https://api.gdc.cancer.gov/cases'

# The 'fields' parameter is passed as a comma-separated string of single names.
fields = [
    "submitter_id",
    "case_id",
    "diagnoses.*",
    "samples"
    ]

fields = ','.join(fields)


filters = {
    "op": "in",
    "content":{
        "field": "cases.project.project_id",
        "value": "TCGA-LIHC"
        }
    }

# With a GET request, the filters parameter needs to be converted
# from a dictionary to JSON-formatted string

params = {
    "filters": json.dumps(filters),
    "fields": fields,
    "format": "JSON",
    "size": "500",
    "return_manifest": True,
    "expand": "diagnoses,samples.*"
    }

response = requests.get(cases_endpt, params = params).json()

if response['data']['pagination']['pages'] != 1:
    print("you need to figure out how to deal w/ pagination")

slide_uuids = []
slides = []
finfo = []


for case in response['data']['hits']:
    for sample in case.get('samples',[]):
        for portion in sample.get('portions',[]):
            for slide in portion.get('slides',[]):
                slides.append({
                    'case_id': case['case_id'],
                    'submitter_id': case['submitter_id'],
                    'sample_type': sample['sample_type'],
                    'sample_type_id': sample['sample_type_id'],
                    'age_at_diagnosis': case['diagnoses'][0]['age_at_diagnosis'],
                    'days_to_birth': case['diagnoses'][0]['days_to_birth'],
                    'days_to_death': case['diagnoses'][0]['days_to_death'],
                    'days_to_last_follow_up': case['diagnoses'][0]['days_to_last_follow_up'],
                    'state': case['diagnoses'][0]['state'],
                    'section_location': slide['section_location'],
                    'percent_normal_cells': slide['percent_normal_cells'],
                    'percent_stromal_cells': slide['percent_stromal_cells'],
                    'percent_tumor_cells': slide['percent_tumor_cells'],
                    'percent_tumor_nuclei': slide['percent_tumor_nuclei'],
                    'section_location': slide['section_location'],
                    'creation_datetime': portion['creation_datetime'],
                    'slide_file_name': '.'.join([slide['submitter_id'], slide['slide_id']]).upper() + '.svs',
                    'slide_id': slide['slide_id']
                })
slides_df = pd.DataFrame(slides)
slides_df.to_csv('slides.csv', index=False)


cases_endpt = 'https://api.gdc.cancer.gov/legacy/files'

# The 'fields' parameter is passed as a comma-separated string of single names.
fields = [
    "files.data_format",
    "files.file_id",
    "files.file_name",
    "files.md5sum"
    ]

fields = ','.join(fields)

filters = {
    "op": "and",
    "content": [
        {
            "op":"=",
            "content":{
                "field":"files.data_format",
                "value":[
                    "SVS"
                ]
            },
        },
        {
            "op":"=",
            "content":{
                "field":"cases.project.project_id",
                "value":[
                    "TCGA-LIHC"
                ]
            }
        }
    ]
}

# With a GET request, the filters parameter needs to be converted
# from a dictionary to JSON-formatted string

params = {
    "filters": json.dumps(filters),
    "fields": fields,
    "format": "JSON",
    "size": "500"
}

response = requests.get(cases_endpt, params = params).json()

if response['data']['pagination']['pages'] != 1:
    print("you need to figure out how to deal w/ pagination")
    print(response['data']['pagination'])

files_df = pd.DataFrame(response['data']['hits'])
manifest_df = files_df[['id','file_name','md5sum','file_size','state']]
manifest_df.columns = ['id','filename','md5','size','state']

manifest_df.to_csv('manifest.txt', sep='\t', index=False)


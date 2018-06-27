import pickle
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
            
    with open('diagnoses_dict.pkl','rb') as f:
        out_dict = pickle.load(f)
    df = pd.DataFrame([out_dict[key]['diagnoses'] for key in out_dict.keys()])
    ipdb.set_trace()

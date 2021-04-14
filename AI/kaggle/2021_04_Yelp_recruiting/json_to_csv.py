import pandas as pd
import json
import numpy as np

def convert_to_csv(fn):
    file = open(fn + '.json', 'r')
    data = [json.loads(line) for line in file]
    file.close()
    
    df = pd.DataFrame(data)
    df.to_csv(fn + '.csv')

if __name__ == '__main__':
    convert_to_csv('yelp_test_set_business')
    convert_to_csv('yelp_test_set_checkin')
    convert_to_csv('yelp_test_set_review')
    convert_to_csv('yelp_test_set_user')

    convert_to_csv('yelp_training_set_business')
    convert_to_csv('yelp_training_set_checkin')
    convert_to_csv('yelp_training_set_review')
    convert_to_csv('yelp_training_set_user')
    

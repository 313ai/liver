from lifelines import CoxPHFitter

import numpy as np

import pandas as pd

from lifelines.utils import k_fold_cross_validation
from lifelines.utils import concordance_index


def main():
    """ """
    # simulating a feature matrix for 100 samples with 50 features
    data = np.random.random((100, 50))
    # simulating time of observations (days) min 10 days, max 2500 days
    observed_time = np.random.randint(10, 2500, (100))
    # simulating event (death) 0 did not occur 1 occured
    observed_event = np.random.randint(0, 2, (100))

    test_data = np.random.random((25, 50))
    test_observed_time = np.random.randint(10, 2500, (25))
    test_observed_event = np.random.randint(0, 2, (25))


    for feature_id, feature_vect in enumerate(data.T):
        dataframe = pd.DataFrame({
            'feature nb{0}'.format(feature_id): feature_vect,
            'event': observed_event,
            'time': observed_time})

        #building a coxph model to see the significance of each independant feature
        cox_model = CoxPHFitter()

        cox_model.fit(dataframe,
                      duration_col='time',
                      event_col='event')

        pvalue = cox_model.summary.p[0]
        print('pvalue: {0} for feature nb: {1}'.format(
            pvalue, feature_id))

        if pvalue > 0.05:
            print('feature nb {0} not overall significant!'.format(feature_id))
            continue

        # test the robustness: score close / higher to 0.7 is a good sign
        scores = k_fold_cross_validation(cox_model,
                                         dataframe,
                                         duration_col='time',
                                         event_col='event',
                                         k=3)

        print('score (mean) (c-index) for {0}'.format(np.mean(scores)))

        # validate the features on the test set
        test_dataframe = pd.DataFrame({
            'feature nb{0}'.format(feature_id): test_data.T[feature_id],
            'event': test_observed_event,
            'time': test_observed_time})

        inferred_time = cox_model.predict_expectation(test_dataframe)

        validation_c_index = concordance_index(
            test_observed_time,
            inferred_time,
            test_observed_event)

        print('validation c-index: {0}'.format(validation_c_index))


if __name__ == '__main__':
    main()

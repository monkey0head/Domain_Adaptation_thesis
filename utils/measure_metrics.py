import datetime
import json


def calc_metric(metric_functor, model, dataset_gen):
    metric = metric_functor()
    for X_batch, y_batch in dataset_gen:
        y_pred = model.predict(X_batch)
        metric(y_batch, y_pred)
    return metric.score


def measure_metric(metric_functor, data_dict, save=False, file_name=None):
    # TODO mb move save to another func
    required_keys = {'topic', 'load_model', 'load_dataset'}
    for i in data_dict:
        assert required_keys == set(i.keys())

    results = {}

    for element in data_dict:
        model = element['load_model']()
        dataset = element['load_dataset']()
        metric_score = calc_metric(metric_functor, model, dataset)
        results[element['topic']] = metric_score

    if save:
        if file_name is None:
            now = datetime.datetime.now()
            now.strftime("%Y-%m-%d-%H:%M:%S")
            file_name = '_'.join([i['topic'] for i in data_dict])
            file_name += '_' + now.strftime("%Y-%m-%d-%H:%M:%S")
        with open(file_name, 'w') as f:
            json.dump(results, f)

    return results

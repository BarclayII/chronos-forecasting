from gluonts.dataset.repository.datasets import get_dataset
from pathlib import Path
from scipy.stats import gmean
from gluonts.time_feature import get_seasonality
import numpy as np
import torch
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

def evaluate(predictor, batch_size=10, num_samples=16):
    gluonts_datasets = [
        "australian_electricity_demand",
        "cif_2016",
        "car_parts_without_missing",
        "covid_deaths",
        "dominick",
        "ercot",
        "ett_small_15min",
        "ett_small_1h",
        "exchange_rate",
        "fred_md",
        "hospital",
        "m1_monthly",
        "m1_quarterly",
        "m1_yearly",
        "m3_monthly",
        "m3_quarterly",
        "m3_yearly",
        "m4_quarterly",
        "m4_yearly",
        "m5",
        #"nn5_daily_with_missing",
        "nn5_daily_without_missing",
        "nn5_weekly",
        "tourism_monthly",
        "tourism_quarterly",
        "tourism_yearly",
        "traffic",
        "weather"
    ]
    # Dictionary to store the loaded datasets
    loaded_datasets = {}

    # Load each dataset
    for dataset_name in gluonts_datasets:
        print(f"Loading dataset: {dataset_name}")
        loaded_datasets[dataset_name] = get_dataset(dataset_name, path=Path('/mnt/efs/gq/gluonts/datasets'))

    print("All datasets loaded successfully.")

    baseline_metrics_dfs = {}
    for dataset_name, dataset in loaded_datasets.items():
        prediction_length = dataset.metadata.prediction_length

        # Split dataset for evaluation
        _, test_template = split(dataset.test, offset=-prediction_length)
        test_data = test_template.generate_instances(prediction_length)

        input_dfs = []
        for i, (train, test) in enumerate(test_data):
            df = pd.DataFrame({'y': train['target'], 'ds': np.arange(len(train['target']))})
            df['unique_id'] = i
            input_dfs.append(df)
        input_df = pd.concat(input_dfs, axis=0)

        sf = StatsForecast(models=[SeasonalNaive(season_length=get_seasonality(dataset.metadata.freq))], freq=1, df=input_df)
        forecast = sf.forecast(h=prediction_length)['SeasonalNaive']
        forecast_samples = forecast.values.reshape((-1, 1, prediction_length))
        
        # Convert forecast samples into gluonts SampleForecast objects
        sample_forecasts = []
        for item, ts in zip(forecast_samples, test_data.input):
            forecast_start_date = ts["start"] + len(ts["target"])
            sample_forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )
        
        # Evaluate
        metrics_df = evaluate_forecasts(
            sample_forecasts,
            test_data=test_data,
            metrics=[
                MASE(),
                MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
            ],
        )
        baseline_metrics_dfs[dataset_name] = metrics_df

    for name, df in baseline_metrics_dfs.items():
        df['name'] = name
    baseline_metrics_df = pd.concat(baseline_metrics_dfs.values(), axis=0)

    metrics_dfs = {}
    for dataset_name, dataset in loaded_datasets.items():
        prediction_length = dataset.metadata.prediction_length

        
        # Split dataset for evaluation
        _, test_template = split(dataset.test, offset=-prediction_length)
        test_data = test_template.generate_instances(prediction_length)
        
        # Generate forecast samples
        forecast_samples = []
        for batch in tqdm(batcher(test_data.input, batch_size=batch_size)):
            context = [torch.tensor(entry["target"]) for entry in batch]
            forecast_samples.append(
                predictor.predict(
                    context,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                ).numpy()
            )
        forecast_samples = np.concatenate(forecast_samples)
        
        # Convert forecast samples into gluonts SampleForecast objects
        sample_forecasts = []
        for item, ts in zip(forecast_samples, test_data.input):
            forecast_start_date = ts["start"] + len(ts["target"])
            sample_forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )
        
        # Evaluate
        metrics_df = evaluate_forecasts(
            sample_forecasts,
            test_data=test_data,
            metrics=[
                MASE(),
                MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
            ],
        )
        metrics_dfs[dataset_name] = metrics_df

    for name, df in metrics_dfs.items():
        df['name'] = name
    metrics_df = pd.concat(metrics_dfs.values(), axis=0)

    all_metrics = pd.merge(baseline_metrics_df, metrics_df, on='name', suffixes=('_baseline', ''))

    mean_mase = gmean(metrics_df['MASE[0.5]'] / baseline_metrics_df['MASE[0.5]'])
    return mean_mase, all_metrics

if __name__ == '__main__':
    from chronos import ChronosPipeline
    import sys
    path = sys.argv[1]
    pipeline = ChronosPipeline.from_pretrained(
        path,
    )
    mean_mase, all_metrics = evaluate(pipeline)
    print(all_metrics)
    print('Mean relative MASE:', mean_mase)

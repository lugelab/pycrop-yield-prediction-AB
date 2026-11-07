
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .kriging import SpatioTemporalResidualKriging
from ..data.utils import select_counties


class RFModelForecast:

    def __init__(
        self,
        savedir=Path("data/models"),
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.3,
        random_state=42,
        use_kriging=True,
        kriging_r_loc_km=100.0,
        kriging_r_year=2.0,
        kriging_nugget_fraction=0.25,
        kriging_neighbors=100,
        min_train_years=3,
        residual_window_years=5,
    ):

        self.savedir = Path(savedir) / "rf"
        self.savedir.mkdir(parents=True, exist_ok=True)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.use_kriging = use_kriging

        self.kriging_r_loc_km = kriging_r_loc_km
        self.kriging_r_year = kriging_r_year
        self.kriging_nugget_fraction = kriging_nugget_fraction
        self.kriging_neighbors = kriging_neighbors

        self.min_train_years = min_train_years
        self.residual_window_years = residual_window_years

    def _new_model(self, random_state):

        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=random_state,
            n_jobs=-1,
        )

    @staticmethod
    def _metrics(actual, predicted):

        return (
            np.sqrt(mean_squared_error(actual, predicted)),
            np.mean(actual - predicted),
            mean_absolute_error(actual, predicted),
        )

    
    @staticmethod
    def _prepare_features(images, time, forecast_steps):
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive")

        if time + forecast_steps > images.shape[2]:
            raise ValueError(
                f"Not enough observations: time={time}, forecast_steps={forecast_steps}, available={images.shape[2]}"
            )

        historical_images = images[:, :, :time, :]

        forecast_images = images[
            :, 7:9, time:time + forecast_steps, :
        ]

        X_historical = historical_images.reshape(
            images.shape[0], -1
        )
        X_forecast = forecast_images.reshape(
            images.shape[0], -1
        )
        X = np.concatenate(
            [X_historical, X_forecast],
            axis=1
        )

        return X.astype(np.float32)

    def _generate_oof_residuals(
        self,
        X,
        y,
        years,
        random_state,
    ):

        unique_years = np.sort(np.unique(years))

        residuals = []
        residual_locations = []
        residual_years = []

        for validation_year in unique_years:

            train_idx = np.where(
                years < validation_year
            )[0]

            validation_idx = np.where(
                years == validation_year
            )[0]

            if len(np.unique(years[train_idx])) < self.min_train_years:
                continue

            fold_model = self._new_model(random_state)

            fold_model.fit(
                X[train_idx],
                y[train_idx],
            )

            fold_predictions = fold_model.predict(
                X[validation_idx]
            )

            fold_residuals = (
                y[validation_idx] - fold_predictions
            )

            residuals.extend(fold_residuals)

            residual_locations.extend(
                self.locations[validation_idx]
            )

            residual_years.extend(
                years[validation_idx]
            )

        if not residuals:
            raise ValueError(
                "No OOF residuals generated. Reduce min_train_years or provide more years."
            )

        residuals = np.asarray(residuals)
        residual_locations = np.asarray(residual_locations)
        residual_years = np.asarray(residual_years)

        if self.residual_window_years is not None:

            latest_year = np.max(residual_years)

            earliest_year = (
                latest_year - self.residual_window_years + 1
            )

            mask = residual_years >= earliest_year

            residuals = residuals[mask]
            residual_locations = residual_locations[mask]
            residual_years = residual_years[mask]

        return (
            residual_locations,
            residual_years,
            residuals,
        )

    
    def run(
        self,
        path_to_histogram=Path("data/img_output/histogram_all_full.npz"),
        pred_years=None,
        times="all",
        num_runs=2,
        target_state=None,
        forecast_steps=1,
    ):
        if not isinstance(forecast_steps, int) or forecast_steps < 1:
            raise ValueError("forecast_steps must be a positive integer")

        with np.load(path_to_histogram) as data:
            images = data["output_image"]
            yields = data["output_yield"]
            years = data["output_year"]
            indices = data["output_index"]
            locations = data["output_locations"]

        states = indices[:, 0].astype(int)
        counties = indices[:, 1].astype(int)

        
        # Select a reproducible subset of counties across all states
        sample_mask, selected_counties = select_counties(
            indices,
            years,
            counties_per_state=8,
            required_years=range(2003, 2017),
            random_state=42,
        )

        images = images[sample_mask]
        yields = yields[sample_mask]
        years = years[sample_mask]
        indices = indices[sample_mask]
        locations = locations[sample_mask]
        states = states[sample_mask]
        counties = counties[sample_mask]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        selected_counties.to_csv(
            self.savedir / f"selected_counties_{timestamp}.csv",
            index=False,
        )

        if len(yields) == 0:
            raise ValueError("No observations found.")

        # Temporal selection
        if times == "all":
            time_points = [32]
        elif times == "realtime":
            time_points = list(range(10, 31, 4))
        else:
            raise ValueError("times must be 'all' or 'realtime'")

        y = yields.astype(float)
        years = years.astype(float)
        locations = locations.astype(float)

        if locations.shape != (len(y), 2):
            raise ValueError("output_locations must have shape (n, 2)")

        if not (
            np.isfinite(images).all()
            and np.isfinite(y).all()
            and np.isfinite(years).all()
            and np.isfinite(locations).all()
        ):
            raise ValueError("Dataset contains NaN or infinite values.")

        if pred_years is None:
            pred_years = range(2009, 2016)
        elif isinstance(pred_years, int):
            pred_years = [pred_years]

        results = []

        for time in time_points:

            for forecast_step in range(1, forecast_steps + 1):

                if time + forecast_step > images.shape[2]:
                    print(
                        f"Skipping time={time}, forecast_step={forecast_step}: out of observations"
                    )
                    continue

                X = self._prepare_features(
                    images,
                    time,
                    forecast_step,
                )

                for predict_year in pred_years:

                    for run_number in range(1, num_runs + 1):

                        train_idx = np.where(years < predict_year)[0]
                        test_idx = np.where(years == predict_year)[0]

                        if len(train_idx) == 0 or len(test_idx) == 0:
                            print("Skipping: insufficient observations.")
                            continue

                        X_train = X[train_idx]
                        X_test = X[test_idx]

                        y_train = y[train_idx]
                        y_test = y[test_idx]

                        year_train = years[train_idx]
                        year_test = years[test_idx]

                        loc_train = locations[train_idx]
                        loc_test = locations[test_idx]

                        self.locations = loc_train

                        random_state = (
                            self.random_state + run_number - 1
                        )
                        if self.use_kriging:

                            (
                                residual_locations,
                                residual_years,
                                residuals,
                            ) = self._generate_oof_residuals(
                                X_train,
                                y_train,
                                year_train,
                                random_state,
                            )

                        model = self._new_model(random_state)

                        model.fit(X_train, y_train)

                        predictions = model.predict(X_test)

                        rmse, me, mae = self._metrics(
                            y_test,
                            predictions,
                        )

                        corrected_predictions = None
                        correction = None

                        kriging_rmse = np.nan
                        kriging_me = np.nan
                        kriging_mae = np.nan

                        if self.use_kriging:


                            kriging = SpatioTemporalResidualKriging(
                                r_loc_km=self.kriging_r_loc_km,
                                r_year=self.kriging_r_year,
                                nugget_fraction=self.kriging_nugget_fraction,
                                n_neighbors=self.kriging_neighbors,
                            )

                            kriging.fit(
                                residual_locations,
                                residual_years,
                                residuals,
                            )

                            correction = kriging.predict(
                                loc_test,
                                year_test,
                            )

                            corrected_predictions = (
                                predictions + correction
                            )

                            (
                                kriging_rmse,
                                kriging_me,
                                kriging_mae,
                            ) = self._metrics(
                                y_test,
                                corrected_predictions,
                            )

                        state_name = (
                            str(target_state)
                            if target_state is not None
                            else "all_states"
                        )

                        results.append({
                            "state": state_name,
                            "year": predict_year,
                            "run_number": run_number,
                            "time_idx": time,
                            "forecast_steps": forecast_step,
                            "train_observations": len(train_idx),
                            "test_observations": len(test_idx),
                            "RMSE": rmse,
                            "ME": me,
                            "MAE": mae,
                            "RMSE_Kriging": kriging_rmse,
                            "ME_Kriging": kriging_me,
                            "MAE_Kriging": kriging_mae,
                        })
                        
                        joblib.dump(
                            model,
                            self.savedir /
                            (
                                f"rf_{state_name}_{predict_year}_"
                                f"{run_number}_time{time}_"
                                f"forecast{forecast_step}.joblib"
                            ),
                        )

                        if self.use_kriging:

                            joblib.dump(
                                kriging,
                                self.savedir /
                                (
                                    f"kriging_{state_name}_{predict_year}_"
                                    f"{run_number}_time{time}_"
                                    f"forecast{forecast_step}.joblib"
                                ),
                            )
                        
                        # Save predictions
                        prediction_data = pd.DataFrame({
                            "State": states[test_idx],
                            "County": counties[test_idx],
                            "Year": years[test_idx],
                            "Longitude": loc_test[:, 0],
                            "Latitude": loc_test[:, 1],
                            "Actual_Yield": y_test,
                            "Predicted_RF": predictions,
                            "RF_Residual": y_test - predictions,
                            "Time": time,
                            "Forecast_Steps": forecast_step,
                        })

                        if self.use_kriging:

                            prediction_data["Kriging_Correction"] = correction

                            prediction_data["Predicted_RF_Kriging"] = (
                                corrected_predictions
                            )

                            prediction_data["Corrected_Residual"] = (
                                y_test - corrected_predictions
                            )

                        prediction_data.to_csv(
                            self.savedir /
                            (
                                f"predictions_{state_name}_{predict_year}_"
                                f"{run_number}_time{time}_"
                                f"forecast{forecast_step}.csv"
                            ),
                            index=False,
                        )

        # Save combined results
        if not results:
            raise ValueError("No models were trained.")

        results_df = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_df.to_csv(
            self.savedir /
            f"rf_oracle_results_{timestamp}.csv",
            index=False,
        )
        print(results_df.to_string(index=False))

        return results_df

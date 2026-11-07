
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .kriging import SpatioTemporalResidualKriging
from ..data.utils import select_counties

class RFModel:

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

    def _generate_oof_residuals(
        self, X, y, years,random_state,
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

        print(
            f"OOF residual observations: {len(residuals)}"
        )

        print(
            f"OOF residual years: "
            f"{np.sort(np.unique(residual_years))}"
        )

        return (
            residual_locations,
            residual_years,
            residuals,
        )

    def run(
        self,
        path_to_histogram=Path(
            "data/img_output/histogram_all_full.npz"
        ),
        pred_years=None,
        times="all",
        num_runs=2,
        target_state=None,
    ):

        print("Loading histogram dataset...")

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
        """
        # State filtering
        if target_state is not None:

            mask = states == int(target_state)

            images = images[mask]
            yields = yields[mask]
            years = years[mask]
            indices = indices[mask]
            locations = locations[mask]
            states = states[mask]
            counties = counties[mask]
        """
        if len(yields) == 0:
            raise ValueError("No observations found.")

        # Temporal selection
        if times == "all":

            time_points = [32]

        elif times == "realtime":
            print("doing realtime")
            time_points = list(range(10, 31, 4))

        else:
            raise ValueError(
                "times must be 'all' or 'realtime'"
            )


        y = yields.astype(float)
        years = years.astype(float)
        locations = locations.astype(float)

        if locations.shape != (len(y), 2):
            raise ValueError(
                "output_locations must have shape (n, 2)"
            )

        if not (
            np.isfinite(images).all()
            and np.isfinite(y).all()
            and np.isfinite(years).all()
            and np.isfinite(locations).all()
        ):
            raise ValueError(
                "Dataset contains NaN or infinite values."
            )

        self.locations = locations

        if pred_years is None:
            pred_years = range(2009, 2016)

        elif isinstance(pred_years, int):
            pred_years = [pred_years]

        results = []

        for time in time_points:

            images_time = images[:, :, :time, :]

            X = images_time.reshape(
                images_time.shape[0], -1
            ).astype(np.float32)
            for predict_year in pred_years:

                train_idx = np.where(years < predict_year)[0]
                test_idx = np.where(years == predict_year)[0]
                if len(train_idx) == 0 or len(test_idx) == 0:
                    print(
                        f"Skipping {predict_year}: insufficient observations."
                    )
                    continue

                X_train = X[train_idx]
                X_test = X[test_idx]

                y_train = y[train_idx]
                y_test = y[test_idx]

                year_train = years[train_idx]
                year_test = years[test_idx]

                loc_train = locations[train_idx]
                loc_test = locations[test_idx]

                for run_number in range(1, num_runs + 1):

                    random_state = (
                        self.random_state + run_number - 1
                    )

                    if self.use_kriging:

                        print("\nGenerating forward-chained residuals...")
                        self.locations = loc_train

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

                    model.fit(
                        X_train,
                        y_train,
                    )

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
                        "train_observations": len(train_idx),
                        "test_observations": len(test_idx),
                        "RMSE": rmse,
                        "ME": me,
                        "MAE": mae,
                        "RMSE_Kriging": kriging_rmse,
                        "ME_Kriging": kriging_me,
                        "MAE_Kriging": kriging_mae,
                        "Time Index": time,
                    })
                    prediction_data = pd.DataFrame({
                        "State": states[test_idx],
                        "County": counties[test_idx],
                        "Year": years[test_idx],
                        "Longitude": loc_test[:, 0],
                        "Latitude": loc_test[:, 1],
                        "Actual_Yield": y_test,
                        "Predicted_RF": predictions,
                        "RF_Residual": y_test - predictions,
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
                        f"predictions_{state_name}_{predict_year}_{run_number}.csv",
                        index=False,
                    )

                    
                    importance = model.feature_importances_

                    num_times = len(time_points)
                    num_bins = images.shape[3]

                    importance_df = pd.DataFrame({
                        "Feature": np.arange(len(importance)),
                        "Importance": importance,
                    })

                    importance_df["Band"] = (
                        importance_df["Feature"]
                        // (num_times * num_bins)
                    )

                    importance_df["Time_Index"] = np.array(
                        time_points
                    )[
                        (importance_df["Feature"] // num_bins)
                        % num_times
                    ]

                    importance_df["Bin"] = (
                        importance_df["Feature"] % num_bins
                    )

                    importance_df.sort_values(
                        "Importance",
                        ascending=False,
                    ).to_csv(
                        self.savedir /
                        f"importance_{state_name}_{predict_year}_{run_number}.csv",
                        index=False,
                    )
                    
        if not results:
            raise ValueError("No models were trained.")

        results_df = pd.DataFrame(results)

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        results_df.to_csv(
            self.savedir /
            f"rf_results_{timestamp}.csv",
            index=False,
        )

        print(results_df.to_string(index=False))

        return results_df

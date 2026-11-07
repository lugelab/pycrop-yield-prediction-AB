import torch
import copy
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from tqdm import tqdm
from datetime import datetime

from .gp import GaussianProcess
from .loss import l1_l2_loss
from ..data.utils import select_counties

class ModelBaseForecast:
    def __init__(
        self,
        model,
        model_weight,
        model_bias,
        model_type,
        savedir,
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.savedir = savedir / model_type
        self.savedir.mkdir(parents=True, exist_ok=True)

        print(f"Using {device.type}")
        if device.type != "cpu":
            model = model.cuda()
        self.model = model
        self.model_type = model_type
        self.model_weight = model_weight
        self.model_bias = model_bias

        self.device = device

        # for reproducability
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.gp = None
        if use_gp:
            self.gp = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b)

    def run(
        self,
        path_to_histogram=Path("data/img_output/histogram_all_full.npz"),
        times="all",
        pred_years=None,
        num_runs=2,
        train_steps=25000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
        target_state=None,
        forecast_steps=0,
    ):

        with np.load(path_to_histogram) as hist:
            images = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_index"]

        
        # Select a reproducible subset of counties across all states.
        state_mask, selected_counties = select_counties(
            indices,
            years,
            counties_per_state=8,
            required_years=range(2003, 2017),
            random_state=42,
        )

        images = images[state_mask]
        locations = locations[state_mask]
        yields = yields[state_mask]
        years = years[state_mask]
        indices = indices[state_mask]

        print(f"Selected counties: {len(selected_counties)}")
        print(f"Selected observations: {len(yields)}")

        # Save selected counties so I can check later that ti used the same
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        selected_counties.to_csv(
            self.savedir / f"selected_counties_{timestamp}.csv",
            index=False,
        )
       
        # to collect results
        years_list, run_numbers, rmse_list, me_list, times_list, forecast_steps_list = [], [], [], [], [], []
        if self.gp is not None:
            rmse_gp_list, me_gp_list = [], []

        if pred_years is None:
            pred_years = range(2009, 2016)
        elif type(pred_years) is int:
            pred_years = [pred_years]

        if times == "all":
            times = [32]
        else:
            times = range(10, 31, 4)

        print("forecast_steps:", forecast_steps, type(forecast_steps))

        if not isinstance(forecast_steps, int) or forecast_steps < 1:
            raise ValueError(
                "forecast_steps must be a positive integer"
            )
        forecast_steps = range(1, forecast_steps+1)

        for pred_year in pred_years:
            for run_number in range(1, num_runs + 1):
                for time in times:
                    for forecast_step in forecast_steps:
                        if time+forecast_step>images.shape[2]:
                            print(
                                f"Skipping time = {time}, forecast_step = {forecast_step}:out of observations"
                            )
                            continue
                        print(
                            f"Training to predict on {pred_year}, Run number {run_number}, Forecast step {forecast_step}, Time {time}"
                        )

                        results = self._run_1_year(
                            images, yields, years,locations,indices,pred_year,time,run_number,
                            train_steps,batch_size,
                            starter_learning_rate,
                            weight_decay,
                            l1_weight,
                            patience,
                            forecast_step
                        )

                        print(
                            f"FINISHED MODEL: Year={pred_year}, Run={run_number}, Forecast={forecast_step}"
                        )

                        years_list.append(pred_year)
                        run_numbers.append(run_number)
                        times_list.append(time)
                        forecast_steps_list.append(forecast_step)

                        if self.gp is not None:
                            rmse, me, rmse_gp, me_gp = results
                            rmse_gp_list.append(rmse_gp)
                            me_gp_list.append(me_gp)
                        else:
                            rmse, me = results
                        rmse_list.append(rmse)
                        me_list.append(me)

        # save results to a csv file
        data = {
            "year": years_list,
            "run_number": run_numbers,
            "time_idx": times_list,
            "RMSE": rmse_list,
            "ME": me_list,
            "forecast_steps": forecast_steps_list,
        }
        if self.gp is not None:
            data["RMSE_GP"] = rmse_gp_list
            data["ME_GP"] = me_gp_list
        results_df = pd.DataFrame(data=data)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_df.to_csv(
            self.savedir / f"{timestamp}.csv",
            index=False
        )
    def _run_1_year(
        self,
        images,
        yields,
        years,
        locations,
        indices,
        predict_year,
        time,
        run_number,
        train_steps,
        batch_size,
        starter_learning_rate,
        weight_decay,
        l1_weight,
        patience,
        forecast_steps,
    ):
        train_data, test_data = self.prepare_arrays(
            images, yields, locations, indices, years, predict_year, time, forecast_steps
        )

        self.reinitialize_model(time=time, forecast_steps=forecast_steps)

        train_scores, val_scores = self._train(
            train_data.images,
            train_data.yields,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
            train_data.forecast_images,
        )

        results = self._predict(
            train_data.images,
            train_data.forecast_images,
            train_data.yields,
            train_data.locations,
            train_data.indices,
            train_data.years,
            test_data.images,
            test_data.forecast_images,
            test_data.yields,
            test_data.locations,
            test_data.indices,
            test_data.years,
            batch_size,
        )
 
        model_information = {
            "state_dict": self.model.state_dict(),
            "val_loss": val_scores["loss"],
            "train_loss": train_scores["loss"],
        }
        for key in results:
            model_information[key] = results[key]

        model_weight = self.model.state_dict()[self.model_weight]
        model_bias = self.model.state_dict()[self.model_bias]

        if self.model.state_dict()[self.model_weight].device != "cpu":
            model_weight, model_bias = model_weight.cpu(), model_bias.cpu()

        model_information["model_weight"] = model_weight.numpy()
        model_information["model_bias"] = model_bias.numpy()

        if self.gp is not None:
            print("Running Gaussian Process!")
            gp_pred = self.gp.run(
                model_information["train_feat"],
                model_information["test_feat"],
                model_information["train_loc"],
                model_information["test_loc"],
                model_information["train_years"],
                model_information["test_years"],
                model_information["train_real"],
                model_information["model_weight"],
                model_information["model_bias"],
            )
            model_information["test_pred_gp"] = gp_pred.squeeze(1)

        filename = f'{predict_year}_{run_number}_{time}__forecast{forecast_steps}_{"gp" if (self.gp is not None) else ""}.pth.tar'
        torch.save(model_information, self.savedir / filename)
        return self.analyze_results(
            model_information["test_real"],
            model_information["test_pred"],
            model_information["test_pred_gp"] if self.gp is not None else None,
        )

    def _train(
        self,
        train_images,
        train_yields,
        train_steps,
        batch_size,
        starter_learning_rate,
        weight_decay,
        l1_weight,
        patience,
        train_forecast,
    ):
        total_size = train_images.shape[0]
        val_size = total_size // 10
        train_size = total_size - val_size
        print(
            f"After split, training on {train_size} examples,validating on {val_size} examples"
        )
        train_dataset, val_dataset = random_split(
            TensorDataset(train_images, train_forecast, train_yields), (train_size, val_size)
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam(
            [pam for pam in self.model.parameters()],
            lr=starter_learning_rate,
            weight_decay=weight_decay,
        )

        num_epochs = int(train_steps / (train_images.shape[0] / batch_size))
        print(f"Training for {num_epochs} epochs")

        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        step_number = 0
        min_loss = np.inf
        best_state = copy.deepcopy(self.model.state_dict())

        if patience is not None:
            epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_train_scores = defaultdict(list)

            for train_x, train_forecast_x, train_y in tqdm(train_dataloader):
                optimizer.zero_grad()
                pred_y = self.model(train_x, forecast_x=train_forecast_x)

                loss, running_train_scores = l1_l2_loss(
                    pred_y, train_y, l1_weight, running_train_scores
                )
                loss.backward()
                optimizer.step()

                train_scores["loss"].append(loss.item())
                step_number += 1

                if step_number in [4000, 20000]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] /= 10

            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )

            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for (
                    val_x,
                    val_forecast_x,
                    val_y,
                ) in tqdm(val_dataloader):
                    val_pred_y = self.model(val_x, forecast_x=val_forecast_x)

                    val_loss, running_val_scores = l1_l2_loss(
                        val_pred_y, val_y, l1_weight, running_val_scores
                    )

                    val_scores["loss"].append(val_loss.item())

            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )
            epoch_val_loss = np.array(running_val_scores["loss"]).mean()

            if epoch_val_loss < min_loss:
                best_state = copy.deepcopy(self.model.state_dict())
                min_loss = epoch_val_loss

                if patience is not None:
                    epochs_without_improvement = 0
            elif patience is not None:
                epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    self.model.load_state_dict(best_state)
                    print("Early stopping")
                    break

        self.model.load_state_dict(best_state)
        return train_scores, val_scores

    def _predict(
        self,
        train_images,
        train_forecast,
        train_yields,
        train_locations,
        train_indices,
        train_years,
        test_images,
        test_forecast,
        test_yields,
        test_locations,
        test_indices,
        test_years,
        batch_size,
    ):

        train_dataset = TensorDataset(
            train_images, train_forecast, train_yields, train_locations, train_indices, train_years
        )

        test_dataset = TensorDataset(
            test_images, test_forecast, test_yields, test_locations, test_indices, test_years
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_forecast_im, train_yield, train_loc, train_idx, train_year in tqdm(
                train_dataloader
            ):
                model_output = self.model(
                    train_im, forecast_x=train_forecast_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["train_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["train_pred"].extend(pred.squeeze(1).tolist())
                results["train_real"].extend(train_yield.squeeze(1).tolist())
                results["train_loc"].append(train_loc.numpy())
                results["train_indices"].append(train_idx.numpy())
                results["train_years"].extend(train_year.tolist())

            for test_im, test_forecast_im, test_yield, test_loc, test_idx, test_year in tqdm(
                test_dataloader
            ):
                model_output = self.model(
                    test_im, forecast_x=test_forecast_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["test_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())
                results["test_loc"].append(test_loc.numpy())
                results["test_indices"].append(test_idx.numpy())
                results["test_years"].extend(test_year.tolist())

        for key in results:
            if key in [
                "train_feat",
                "test_feat",
                "train_loc",
                "test_loc",
                "train_indices",
                "test_indices",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    def prepare_arrays(
        self, images, yields, locations, indices, years, predict_year, time, forecast_steps=0,
    ):
        train_idx = np.nonzero(years < predict_year)[0]
        test_idx = np.nonzero(years == predict_year)[0]

        train_images, test_images = self._normalize(images[train_idx], images[test_idx])

        train_historical = train_images[:, :, :time, :]
        test_historical = test_images[:, :, :time, :]

        train_forecast = train_images[
            :, 7:9, time:time + forecast_steps, :
        ]

        test_forecast = test_images[
            :, 7:9, time:time + forecast_steps, :
        ]

        if train_forecast.size == 0:
            print("WARNING: Training forecast is empty")

        if test_forecast.size == 0:
            print("WARNING: Testing forecast is empty")

        print(
            f"Train set size: {train_idx.shape[0]}, Test set size: {test_idx.shape[0]}"
        )

        Data = namedtuple("Data", ["images", "forecast_images", "yields", "locations", "indices", "years"])

        train_data = Data(
            images=torch.as_tensor(
                train_historical, device=self.device
            ).float(),

            forecast_images=torch.as_tensor(
                train_forecast, device=self.device
            ).float(),

            yields=torch.as_tensor(
                yields[train_idx], device=self.device
            ).float().unsqueeze(1),

            locations=torch.as_tensor(locations[train_idx]),
            indices=torch.as_tensor(indices[train_idx]),
            years=torch.as_tensor(years[train_idx]),
        )

        test_data = Data(
            images=torch.as_tensor(
                test_historical, device=self.device
            ).float(),

            forecast_images=torch.as_tensor(
                test_forecast, device=self.device
            ).float(),

            yields=torch.as_tensor(
                yields[test_idx], device=self.device
            ).float().unsqueeze(1),

            locations=torch.as_tensor(locations[test_idx]),
            indices=torch.as_tensor(indices[test_idx]),
            years=torch.as_tensor(years[test_idx]),
        )

        return train_data, test_data

    @staticmethod
    def _normalize(train_images, val_images):
        mean = np.mean(train_images, axis=(0, 2, 3))

        train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        return train_images, val_images

    @staticmethod
    def analyze_results(true, pred, pred_gp):
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        me = np.mean(true - pred)

        if pred_gp is not None:
            rmse_gp = np.sqrt(np.mean((true - pred_gp) ** 2))
            me_gp = np.mean(true - pred_gp)
            return rmse, me, rmse_gp, me_gp
        return rmse, me

    def reinitialize_model(self, time=None):
        raise NotImplementedError

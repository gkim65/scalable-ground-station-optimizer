#!/usr/bin/env python
"""
Ground station optimization for maximizing data downlink for Capella spacecraft using only KSAT ground stations.

This example demonstrates how to create a ground station optimization problem to maximize the total data downlinked over
a mission period specifically for a specific constellation and ground station provider. In this example, we use the
Capella constellation and KSAT ground station provider.
"""

import logging
import datetime
from rich.console import Console

from gsopt.milp_objectives import *
from gsopt.milp_constraints import *
from gsopt.milp_optimizer import MilpOptimizer, get_optimizer
from gsopt.models import OptimizationWindow
from gsopt.scenarios import ScenarioGenerator
from gsopt.utils import filter_warnings, download_earth_data
import os
import brahe as bh
from gsopt.ephemeris import get_latest_celestrak_tles
import random
import numpy as np
import pandas as pd
import kmedoids


# WandB & Hydra & json
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
import shutil
import copy

def scenario_gen(cfg, opt_window, run_name, all_true):

    # Create Rich console for pretty printing
    console = Console()

    # OPTIMIZER SELECTION
    optimizer_type = get_optimizer(cfg.method.optimizer)  # 'scip', 'cbc', 'glpk', or 'gurobi'
 
    # Create a scenario generator
    scengen = ScenarioGenerator(opt_window, seed=cfg.debug.randseed)

    # Hack to override the default scenario datarates to be fixed
    scengen._sat_datarate_ranges['default'] = (cfg.scenario.datarate, cfg.scenario.datarate)
    scengen._provider_datarate['default'] = (cfg.scenario.datarate, cfg.scenario.datarate)

    if cfg.scenario.constellations == "WALKER":
        scengen.add_walker_constellation(
            epoch= bh.Epoch(cfg.start_epoch.year, 
                        cfg.start_epoch.month, 
                        cfg.start_epoch.day, 
                        cfg.start_epoch.hour, 
                        cfg.start_epoch.minute, 
                        cfg.start_epoch.second),
            altitude_km=cfg.walker.altitude,
            eccentricity=cfg.walker.eccentricity,
            inclination=cfg.walker.inclination,
            num_planes=cfg.walker.num_planes,
            sats_per_plane= cfg.walker.sats_perplane,
            argp=cfg.walker.argp,
            norad_start=cfg.walker.norad_start,
            star =cfg.walker.star,
        )
    else:
        scengen.add_constellation(cfg.scenario.constellations)

    if run_name ==  "KMediods_IP":
        files = os.listdir('data/selectedStations')
        json_files = [f for f in files if f.endswith('.json')]
        for provider in json_files:
            scengen.add_provider_selected(provider)    
    else:
        if cfg.scenario.providers == "all":
            providers = ["ksat", "atlas", "aws", "azure", "leaf", "ssc", "viasat"]
            for provider in providers:
                scengen.add_provider(f'{provider}.json')    
        else:
            # Add only KSAT provider
            scengen.add_provider(f'{cfg.scenario.providers}.json')

    # Generate a problem instance
    # scen = scengen.sample_scenario()

    if all_true:
        scenarios = [scengen.sample_scenario()]
    else:
        scenarios = scengen.iter_sample_scenario(chunks=cfg.scenario.chunks)

    # Delete all contents in 'output' directory if it exists
    if not os.path.exists("output"):
        os.makedirs("output")
    else:
        for filename in os.listdir("output"):
            file_path = os.path.join("output", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


    # Runnning for every individual satellite scenario
    for i,scen in enumerate(scenarios):
        
        ########## Configuration files: ##########
        #Ensure unique project names every time
        proj_name = cfg.setup.name+"_"+cfg.setup.method+"_"+cfg.setup.objective+"_"+ cfg.scenario.providers+"="+str(cfg.setup.gs_num)+"_"+cfg.scenario.constellations+"="+str(cfg.walker.num_planes)

        if run_name:
            run = wandb.init(entity=cfg.wandb.entity, project=proj_name, name = run_name)
        else:
            run = wandb.init(entity=cfg.wandb.entity, project=proj_name)
            

        config_dict = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

        wandb.config.update(config_dict)

        # Display the generated scenario
        console.print(scen)

        # Create a IP optimizer from the scenario
        optimizer = MilpOptimizer.from_scenario(scen, optimizer=optimizer_type)

        # Save plot of all stations
        optimizer.save_plot('output/capella_ksat_all_stations.png')

        # Compute contacts
        optimizer.compute_contacts()

        # Setup the optimization problem to maximize data downlink
        if cfg.setup.objective == "max_data":
            optimizer.set_objective(MaxDataDownlinkObjective())
        elif cfg.setup.objective == "minmean_gap":
            optimizer.set_objective(MinMeanContactGapObjective())
        elif cfg.setup.objective == "minmax_gap":
            optimizer.set_objective(MinMaxContactGapObjective())
        elif cfg.setup.objective == "mintotal_gap":
            optimizer.set_objective(MinTotalContactGapObjective())
        else:
            optimizer.set_objective(MaxDataDownlinkObjective())

        # Add Constraints
        optimizer.add_constraints([
            MinContactDurationConstraint(min_duration=180.0),  # Minimum 3 minute contact duration
            SatelliteContactExclusionConstraint(),  # This constraint should always be added to avoid self-interference
            StationNumberConstraint(minimum = cfg.setup.gs_num,maximum = cfg.setup.gs_num),
        ])

        # Solve the optimization problem
        optimizer.solve()

        # Display results
        console.print(optimizer)
        
        text = str(scen.opt_window.opt_start)+"___"+str(scen.opt_window.opt_start)
        # Optional: write the solution to a file
        optimizer.write_solution(f'output/capella_ksat_optimization_{text}.json')

        # Plot and save the selected ground stations
        optimizer.save_plot(f'output/capella_ksat_selected_stations{text}.png')


        with open(f'output/capella_ksat_optimization_{text}.json', 'r') as json_file:
            optimization_results = json.load(json_file)

            # Log the entire JSON content into WandB configurations
            run.config.update(optimization_results)

            # Extract key metrics from the JSON file and log them as summary metrics
            if 'statistics' in optimization_results:
                for key, value in optimization_results['statistics'].items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            wandb.summary[f"{key}.{sub_key}"] = sub_value
                    else:
                        wandb.summary[key] = value

            # Example: Log specific fields if they exist
            if 'selected_stations' in optimization_results:
                coords_list = []
                groups = []
                providers = []
                stations = []
                all_stations = optimization_results['providers'][0]['features']
                station_name_to_coordinates = {station['properties']['name']: station['geometry']['coordinates'] for station in all_stations}
                for station in optimization_results['selected_stations']:
                    name = station['name']
                    coords = station_name_to_coordinates.get(name)
                    if coords:
                        coords_list.append([station['name'], station['provider'], float(coords[0]),float(coords[1]), (float(coords[0]),float(coords[1]))])
                        groups.append(text+"sat_chunk"+str(i))  # or f'optimization_{i}' if you want string labels
                # Convert to DataFrame for easier handling
                df = pd.DataFrame(coords_list, columns=['name', 'provider', 'longitude', 'latitude', 'Coord'])
                df['Group'] = groups
                table = wandb.Table(dataframe=df)
                wandb.log({"coords_list": table})
                wandb.summary['selected_stations'] = optimization_results['selected_stations']

            if 'solver_status' in optimization_results:
                wandb.summary['solver_status'] = optimization_results['solver_status']

        wandb.save("output/*")
        run.finish()

    return coords_list, groups

def save_selected_json(matched_stations):

    # Organize matches by provider
    provider_to_names = {}
    for provider, name in matched_stations:
        provider_to_names.setdefault(provider, set()).add(name)

    # Delete all contents in 'data/selectedStations' directory if it exists
    if not os.path.exists("data/selectedStations"):
        os.makedirs("data/selectedStations")
    else:
        for filename in os.listdir("data/selectedStations"):
            file_path = os.path.join("data/selectedStations", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)



    for provider, names in provider_to_names.items():
        # Read the provider's stations JSON file
        input_path = f"data/groundstations/{provider.lower()}.json"
        output_path = f"data/selectedStations/{provider.lower()}.json"
        with open(input_path, "r") as f:
            full_geojson = json.load(f)
        
        # Filter only features matching selected stations
        selected_features = [
            feat for feat in full_geojson["features"]
            if feat["properties"]["name"] in names
        ]
        
        selected_geojson = {
            "type": "FeatureCollection",
            "features": selected_features
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(selected_geojson, f, indent=2)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main_ip(cfg : DictConfig) -> None:

    # Set seed to ensure consistency
    random.seed(cfg.debug.randseed)
    np.random.seed(cfg.debug.randseed)

    filter_warnings()

    if cfg.debug.txtUpdate:
        # Only bring back if we want to use new data
        download_earth_data()
        get_latest_celestrak_tles()

    # Set up logging
    logging.basicConfig(
        datefmt='%Y-%m-%dT%H:%M:%S',
        format='%(asctime)s.%(msecs)03dZ %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        level=logging.INFO
    )

    logger = logging.getLogger(__name__)

    # Create Rich console for pretty printing
    console = Console()

    # OPTIMIZER SELECTION
    optimizer_type = get_optimizer(cfg.method.optimizer)  # 'scip', 'cbc', 'glpk', or 'gurobi'

    # Define the optimization window
    opt_time = datetime.datetime(cfg.start_epoch.year,
                                    cfg.start_epoch.month,
                                    cfg.start_epoch.day,
                                    cfg.start_epoch.hour,
                                    cfg.start_epoch.minute, 
                                    cfg.start_epoch.second, 
                                    cfg.start_epoch.partSecond, 
                                    tzinfo=datetime.timezone.utc)
    current_start = opt_time

    all_coords = []
    all_groups = []
    opt_window_list = []
    while current_start < (opt_time + datetime.timedelta(hours=cfg.opt.full_length * 24)):
        opt_end = current_start + datetime.timedelta(hours=cfg.opt.window_length)
        if opt_end > opt_time + datetime.timedelta(hours=cfg.opt.full_length * 24):
            opt_end = opt_time + datetime.timedelta(hours=cfg.opt.full_length * 24)
        
        opt_window = OptimizationWindow(
            current_start,
            opt_end,
            current_start,
            opt_end,
        )
        opt_window_list.append(opt_window)
        current_start += datetime.timedelta(hours=cfg.opt.window_length - cfg.opt.overlap)


    for window,opt_window in enumerate(opt_window_list):
        coords_list, groups = scenario_gen(cfg, opt_window, "window-"+str(window), False)
        all_coords.extend(coords_list)
        all_groups.extend(groups)
    
    # Full IP:
    scenario_gen(cfg, 
                 OptimizationWindow(
                    opt_time,
                    opt_time + datetime.timedelta(days=cfg.opt.full_length),
                    opt_time,
                    opt_time + datetime.timedelta(days=cfg.opt.full_length)
                ), 
                "IP_true_solution",
                True)
    
    # Kmediods
    if cfg.setup.method == "KMediods":

        # Convert to DataFrame for easier handling
        df_all_og = pd.DataFrame(all_coords, columns=['name', 'provider', 'longitude', 'latitude', 'Coord'])
        df_all_og['Group'] = all_groups
        proj_name = cfg.setup.name+"_"+cfg.setup.method+"_"+cfg.setup.objective+"_"+ cfg.scenario.providers+"="+str(cfg.setup.gs_num)+"_"+cfg.scenario.constellations+"="+str(cfg.walker.num_planes)

        run = wandb.init(entity=cfg.wandb.entity, project=proj_name, name="KclusterList")
        run.config.update(c)

        table = wandb.Table(dataframe=df_all_og)
        wandb.log({"all_cluster_points": table})

        run.finish()
        for seed in range(cfg.setup.trials):
            df_all = copy.deepcopy(df_all_og)
            km = kmedoids.KMedoids(cfg.setup.gs_num, method='fasterpam',metric="euclidean", random_state=cfg.debug.randseed+seed)
            c = km.fit(np.vstack(df_all['Coord'].values))

            # getting the IP based contact windows now:

            # Create a tuple version of 'Coord' for comparison
            df_all['Coord_tuple'] = df_all['Coord'].apply(lambda x: tuple(x))

            # Get the medoid coordinates as tuples
            medoid_tuples = [tuple(coord) for coord in c.cluster_centers_]

            # Find matching rows
            matched = df_all[df_all['Coord_tuple'].isin(medoid_tuples)][['name', 'provider', 'longitude', 'latitude']]

            # Collect all matched stations as (provider, name) pairs
            matched_stations = matched[['provider', 'name']].values.tolist()
            save_selected_json(matched_stations=matched_stations)


            scenario_gen(cfg, 
                    OptimizationWindow(
                        opt_time,
                        opt_time + datetime.timedelta(days=cfg.opt.full_length),
                        opt_time,
                        opt_time + datetime.timedelta(days=cfg.opt.full_length)
                    ), 
                    "KMediods_IP",
                    True)

if __name__ == "__main__":
    main_ip()

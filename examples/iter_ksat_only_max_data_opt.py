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
from gsopt.utils import filter_warnings
import os
import brahe as bh
from gsopt.ephemeris import get_latest_celestrak_tles



filter_warnings()


def download_earth_data(filename):
    # Here we can download the latest Earth orientation data and load it.

    # if not os.path.exists(filename):
    print("Downloading latest earth orientation data...")
    if not os.path.exists("data"):
        os.makedirs("data")
    bh.utils.download_iers_bulletin_ab("./data")
    print("Download complete")

# print(os.path.exists('data/iau2000A_finals_ab.txt'))
# print("YOOOO")
# download_earth_data('data/iau2000A_finals_ab.txt')
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
optimizer_type = get_optimizer('gurobi')  # 'scip', 'cbc', 'glpk', or 'gurobi'

# Define the optimization window
opt_start = datetime.datetime.now(datetime.timezone.utc)
opt_end = opt_start + datetime.timedelta(days=7) # This is the full mission window. Normally 365+ days. Set to 7 days so that the constant is 1.

# The simulation window is the duration over which we want to simulate contacts
# This is shorter than or equal to the optimization window
sim_start = opt_start
sim_end = sim_start + datetime.timedelta(days=7)

opt_window = OptimizationWindow(
    opt_start,
    opt_end,
    sim_start,
    sim_end
)

# Create a scenario generator
scengen = ScenarioGenerator(opt_window)

# Hack to override the default scenario datarates to be fixed
scengen._sat_datarate_ranges['default'] = (1.2e9, 1.2e9)
scengen._provider_datarate['default'] = (1.2e9, 1.2e9)

# Add only Capella constellation
scengen.add_constellation('CAPELLA')

# scengen.add_constellation('ICEYE')

# Add only KSAT provider
scengen.add_provider('ksat.json')

# Generate a problem instance
scenarios = scengen.iter_sample_scenario() # NEW CHANGE FROM GRACE

scena = scengen.sample_scenario() # NEW CHANGE FROM GRACE

if not os.path.exists("output"):
    os.makedirs("output")

for i,scen in enumerate(scenarios):
    # Display the generated scenario
    console.print(scen)

    # Create a MILP optimizer from the scenario
    optimizer = MilpOptimizer.from_scenario(scen, optimizer=optimizer_type)

    # Save plot of all stations
    optimizer.save_plot('./output/capella_ksat_all_stations.png')

    # Compute contacts
    optimizer.compute_contacts()

    # Setup the optimization problem to maximize data downlink
    optimizer.set_objective(
        MaxDataDownlinkObjective()
    )

    # Add Constraints
    optimizer.add_constraints([
        MinContactDurationConstraint(min_duration=180.0),  # Minimum 3 minute contact duration
        # StationContactExclusionConstraint(),
        SatelliteContactExclusionConstraint(),  # This constraint should always be added to avoid self-interference
        MaxStationsConstraint(num_stations=5) # At most 5 stations
    ])

    # Solve the optimization problem
    optimizer.solve()

    # Display results
    console.print(optimizer)

    if i == len(scenarios)-1:
        text = "global"
    else:
        text = i
    # Optional: write the solution to a file
    optimizer.write_solution(f'./output/capella_ksat_optimization_{text}.json')

    # Plot and save the selected ground stations
    optimizer.save_plot(f'./output/capella_ksat_selected_stations{text}.png')
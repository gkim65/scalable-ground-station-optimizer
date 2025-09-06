"""
Module containing different objective functions for MILP optimization
"""

from abc import abstractmethod, ABCMeta
from itertools import groupby

import pyomo.kernel as pk

from gsopt.milp_core import ProviderNode, StationNode, ContactNode, SatelliteNode
from gsopt.models import OptimizationWindow
from gsopt.utils import time_milp_generation


class GSOptObjective(metaclass=ABCMeta):
    """
    Abstract class for the objective function of the MILP optimization.

    Enforces the implementation of the _generate_objective method.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.obj = pk.objective()

        self.obj.expr = 0

    @abstractmethod
    def _generate_objective(self):
        pass

    def dict(self):
        return {
            'type': self.__class__.__name__,
            'args': self.kwargs
        }


class MinCostObjective(pk.block, GSOptObjective):
    """
    Objective function for the MILP optimization that minimizes the total cost (capital and operational) of the
    ground station provider over the optimization period.
    """

    def __init__(self, **kwargs):
        pk.block.__init__(self)
        GSOptObjective.__init__(self, **kwargs)

        # Set objective direction
        self.obj.sense = pk.minimize

    @time_milp_generation
    def _generate_objective(self, provider_nodes: dict[str, ProviderNode] | None = None,
                            station_nodes: dict[str, StationNode] | None = None,
                            contact_nodes: dict[str, ContactNode] | None = None,
                            station_satellite_nodes: dict[(str, str), pk.variable] | None = None,
                            opt_window: OptimizationWindow | None = None, **kwargs):
        """
        Generate the objective function.
        """

        # Add provider costs
        for pn_id, pn in provider_nodes.items():
            self.obj.expr += pn.model.integration_cost * provider_nodes[pn_id].var

        # Add station costs
        for sn_id, sn in station_nodes.items():
            self.obj.expr += sn.model.setup_cost * station_nodes[sn_id].var

            # Add monthly station costs, normalized to the optimization period
            self.obj.expr += (12 * opt_window.T_opt) / (365.25 * 86400.0 * opt_window.T_sim) * sn.model.monthly_cost * station_nodes[sn_id].var

            # Add satellite licensing costs for the station
            for key in filter(lambda x: x[0] == sn_id, station_satellite_nodes.keys()):
                self.obj.expr += sn.model.per_satellite_license_cost * station_satellite_nodes[key]

        # Add contact costs
        for cn_id, cn in contact_nodes.items():
            self.obj.expr += opt_window.T_opt / opt_window.T_sim * (cn.model.t_duration * cn.model.cost_per_minute + cn.model.cost_per_pass) * contact_nodes[cn_id].var


class MaxDataDownlinkObjective(pk.block, GSOptObjective):
    """
    Objective function for the MILP optimization that maximizes the total data downlinked by the constellation over the
    optimization period.
    """

    def __init__(self, **kwargs):
        pk.block.__init__(self)
        GSOptObjective.__init__(self, **kwargs)

        # Set objective direction
        self.obj.sense = pk.maximize

    @time_milp_generation
    def _generate_objective(self, provider_nodes: dict[str, ProviderNode] | None = None,
                            station_nodes: dict[str, StationNode] | None = None,
                            contact_nodes: dict[str, ContactNode] | None = None,
                            opt_window: OptimizationWindow | None = None, **kwargs):
        """
        Generate the objective function.
        """

        for cn in contact_nodes.values():
            self.obj.expr += cn.var * cn.model.data_volume * opt_window.T_opt / opt_window.T_sim


# class MinMaxContactGapObjective(pk.block, GSOptObjective):
#     """
#     Objective function for the MILP optimization that minimizes the maximum gap between contacts across all satellites
#     in the constellation over the optimization period.
#     """

#     def __init__(self, **kwargs):
#         pk.block.__init__(self)
#         GSOptObjective.__init__(self, **kwargs)

#         # Set objective direction
#         self.obj.sense = pk.minimize

#         # Initialize constraints required to implement the objective
#         self.constraints = pk.constraint_list()

#     @time_milp_generation
#     def _generate_objective(self, provider_nodes: dict[str, ProviderNode] | None = None,
#                             station_nodes: dict[str, StationNode] | None = None,
#                             contact_nodes: dict[str, ContactNode] | None = None,
#                             opt_window: OptimizationWindow | None = None, **kwargs):
#         """
#         Generate the objective function.
#         """

#         # Group contacts by satellite
#         contact_nodes_by_satellite = sorted(contact_nodes.values(), key=lambda cn: cn.satellite.id)

#         self.variable_dict = pk.variable_dict()

#         # Create auxiliary variable for the max gap across all satellites and contacts
#         self.variable_dict['max_gap'] = pk.variable(value=0.0, domain=pk.NonNegativeReals)

#         # Set objective to minimize the maximum gap
#         self.obj.expr = self.variable_dict['max_gap']

#         for sat_id, sat_contacts in groupby(contact_nodes_by_satellite, lambda cn: cn.satellite.id):
#             # Sort contacts by start time
#             sat_contacts = list(sorted(sat_contacts, key=lambda cn: cn.model.t_start))

#             # Force at least one contact scheduled for this satellite
#             contact_vars = [contact_nodes[cn.id].var for cn in sat_contacts]
#             self.constraints.append(pk.constraint(sum(contact_vars) >= 2))


#             # For each contact, create an auxiliary variable for the next scheduled task
#             for i, cn_i in enumerate(sat_contacts[0:len(sat_contacts) - 1]):

#                 # Working expression for the next scheduled contact
#                 expr = pk.expression(0)

#                 for j, cn_j in enumerate(filter(lambda cn: cn.model.t_start > cn_i.model.t_end, sat_contacts)):
#                     # Auxiliary variable if contact j is the next scheduled after contact i
#                     self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)] = pk.variable(value=0, domain=pk.Binary)

#                     expr += self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)]

#                     # Constraints to ensure that if the auxiliary variable is 1, then both x_i and x_j are 1
#                     self.constraints.append(pk.constraint(
#                         self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)] <= contact_nodes[cn_i.id].var))
#                     self.constraints.append(pk.constraint(
#                         self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)] <= contact_nodes[cn_j.id].var))

#                     # Add constraint to ensure that the associated scheduled gap is less than the maximum
#                     self.constraints.append(pk.constraint((cn_j.model.t_start - cn_i.model.t_end) * self.variable_dict[
#                         (sat_id, cn_i.model.id, cn_j.model.id)] <= self.variable_dict['max_gap']))
#                     # print(self.constraints[-1].expr)


#                 # Add constraint that only one of the auxiliary variables can be 1 if the contact node is scheduled
#                 self.constraints.append(pk.constraint(expr == contact_nodes[cn_i.id].var))
class MinMaxContactGapObjective(pk.block, GSOptObjective):
    """
    Objective function for the MILP optimization that minimizes the maximum gap between contacts across all satellites
    in the constellation over the optimization period. Includes the optimization window start and end boundaries.
    """

    def __init__(self, **kwargs):
        pk.block.__init__(self)
        GSOptObjective.__init__(self, **kwargs)

        # Set objective direction
        self.obj.sense = pk.minimize

        # Initialize constraints required to implement the objective
        self.constraints = pk.constraint_list()

    @time_milp_generation
    def _generate_objective(self, provider_nodes: dict[str, ProviderNode] | None = None,
                            station_nodes: dict[str, StationNode] | None = None,
                            contact_nodes: dict[str, ContactNode] | None = None,
                            opt_window: OptimizationWindow | None = None, **kwargs):
        """
        Generate the objective function.
        """

        # Group contacts by satellite
        contact_nodes_by_satellite = sorted(contact_nodes.values(), key=lambda cn: cn.satellite.id)

        self.variable_dict = pk.variable_dict()

        # Create auxiliary variable for the max gap across all satellites and contacts
        self.variable_dict['max_gap'] = pk.variable(value=0.0, domain=pk.NonNegativeReals)

        # Set objective to minimize the maximum gap
        self.obj.expr = self.variable_dict['max_gap']

        for sat_id, sat_contacts in groupby(contact_nodes_by_satellite, lambda cn: cn.satellite.id):
            # Sort contacts by start time
            sat_contacts = list(sorted(sat_contacts, key=lambda cn: cn.model.t_start))

            # Force at least one contact scheduled for this satellite
            # NOTE: you may want to change >= 2 → >= 1 if single contacts should be allowed
            contact_vars = [contact_nodes[cn.id].var for cn in sat_contacts]
            self.constraints.append(pk.constraint(sum(contact_vars) >= 1))

            # ---- Include opt_window start → first contact gap ----
            for cn_first in sat_contacts:
                self.variable_dict[(sat_id, "opt_start", cn_first.model.id)] = pk.variable(value=0, domain=pk.Binary)

                # If this is the first scheduled contact, enforce consistency
                self.constraints.append(pk.constraint(
                    self.variable_dict[(sat_id, "opt_start", cn_first.model.id)] <= contact_nodes[cn_first.id].var
                ))

                # Gap constraint from opt_start to first contact start
                self.constraints.append(pk.constraint(
                    (cn_first.model.t_start - opt_window.opt_start) *
                    self.variable_dict[(sat_id, "opt_start", cn_first.model.id)]
                    <= self.variable_dict['max_gap']
                ))

            # Ensure exactly one "first" contact if any scheduled
            self.constraints.append(pk.constraint(
                sum(self.variable_dict[(sat_id, "opt_start", cn.model.id)] for cn in sat_contacts) == 1
            ))

            # ---- Include last contact → opt_window end gap ----
            for cn_last in sat_contacts:
                self.variable_dict[(sat_id, cn_last.model.id, "opt_end")] = pk.variable(value=0, domain=pk.Binary)

                # If this is the last scheduled contact, enforce consistency
                self.constraints.append(pk.constraint(
                    self.variable_dict[(sat_id, cn_last.model.id, "opt_end")] <= contact_nodes[cn_last.id].var
                ))

                # Gap constraint from last contact end to opt_end
                self.constraints.append(pk.constraint(
                    (opt_window.opt_end - cn_last.model.t_end) *
                    self.variable_dict[(sat_id, cn_last.model.id, "opt_end")]
                    <= self.variable_dict['max_gap']
                ))

            # Ensure exactly one "last" contact if any scheduled
            self.constraints.append(pk.constraint(
                sum(self.variable_dict[(sat_id, cn.model.id, "opt_end")] for cn in sat_contacts) == 1
            ))

            # ---- Original internal gap constraints ----
            for i, cn_i in enumerate(sat_contacts[0:len(sat_contacts) - 1]):
                expr = pk.expression(0)

                for j, cn_j in enumerate(filter(lambda cn: cn.model.t_start > cn_i.model.t_end, sat_contacts)):
                    self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)] = pk.variable(value=0, domain=pk.Binary)

                    expr += self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)]

                    # Constraints to ensure if aux var = 1 → both contacts are scheduled
                    self.constraints.append(pk.constraint(
                        self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)] <= contact_nodes[cn_i.id].var))
                    self.constraints.append(pk.constraint(
                        self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)] <= contact_nodes[cn_j.id].var))

                    # Gap constraint between consecutive contacts
                    self.constraints.append(pk.constraint(
                        (cn_j.model.t_start - cn_i.model.t_end) *
                        self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)]
                        <= self.variable_dict['max_gap']
                    ))

                # Add constraint that only one "next contact" can follow if scheduled
                self.constraints.append(pk.constraint(expr == contact_nodes[cn_i.id].var))


class MinMeanContactGapObjective(pk.block, GSOptObjective):
    """
    Objective function for the MILP optimization that minimizes the maximum gap between contacts across all satellites
    in the constellation over the optimization period.
    """

    def __init__(self, **kwargs):
        pk.block.__init__(self)
        GSOptObjective.__init__(self, **kwargs)

        # Set objective direction
        self.obj.sense = pk.minimize
        self.variable_dict = pk.variable_dict()

        # Initialize constraints required to implement the objective
        self.constraints = pk.constraint_list()

    @time_milp_generation
    def _generate_objective(self, provider_nodes: dict[str, ProviderNode] | None = None,
                            station_nodes: dict[str, StationNode] | None = None,
                            contact_nodes: dict[str, ContactNode] | None = None,
                            satellite_nodes: dict[str, SatelliteNode] | None = None,                            
                            opt_window: OptimizationWindow | None = None, **kwargs):
        """
        Generate the objective function.
        """

        # Group contacts by satellite
        contact_nodes_by_satellite = sorted(contact_nodes.values(), key=lambda cn: cn.satellite.id)
        self.satellite_gap_vars = dict()      # gap variables per satellite
        self.satellite_num_gaps = dict()      # gap counts per satellite
        self.satellite_mean_gap = dict()      # mean gap variable per satellite

        for sat in satellite_nodes.values():
            self.satellite_gap_vars[sat.id] = []
            self.satellite_num_gaps[sat.id] = pk.variable(domain=pk.NonNegativeReals)
            self.satellite_mean_gap[sat.id] = pk.variable(domain=pk.NonNegativeReals)


        for sat_id, sat_contacts in groupby(contact_nodes_by_satellite, lambda cn: cn.satellite.id):
            sat_contacts = sorted(sat_contacts, key=lambda cn: cn.model.t_start)
            for i, cn_i in enumerate(sat_contacts[:-1]):
                expr = pk.expression(0)
                for cn_j in filter(lambda cn: cn.model.t_start > cn_i.model.t_end, sat_contacts):
                    pair_var = pk.variable(domain=pk.Binary)
                    self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)] = pair_var
                    expr += pair_var

                    self.constraints.append(pk.constraint(pair_var <= contact_nodes[cn_i.id].var))
                    self.constraints.append(pk.constraint(pair_var <= contact_nodes[cn_j.id].var))

                    gap_var = pk.variable(domain=pk.NonNegativeReals)
                    self.satellite_gap_vars[sat_id].append(gap_var)

                    self.constraints.append(pk.constraint(gap_var >= (cn_j.model.t_start - cn_i.model.t_end) * pair_var))
                    self.constraints.append(pk.constraint(gap_var <= (cn_j.model.t_start - cn_i.model.t_end) * pair_var))

                    self.satellite_num_gaps[sat_id] += pair_var

                self.constraints.append(pk.constraint(expr == contact_nodes[cn_i.id].var))

        for sat in satellite_nodes.values():
            total_gap = pk.expression(sum(self.satellite_gap_vars[sat.id]))
            self.constraints.append(pk.constraint(
                self.satellite_mean_gap[sat.id] * self.satellite_num_gaps[sat.id] == total_gap
            ))
        
        mean_gap_sum = pk.expression(sum(self.satellite_mean_gap.values()))
        self.obj.expr = mean_gap_sum / len(satellite_nodes)


class MinTotalContactGapObjective(pk.block, GSOptObjective):
    """
    Objective function for the MILP optimization that minimizes the total gap time between contacts
    across all satellites in the constellation over the optimization period.
    """

    def __init__(self, **kwargs):
        pk.block.__init__(self)
        GSOptObjective.__init__(self, **kwargs)
        self.obj.sense = pk.minimize
        self.variable_dict = pk.variable_dict()
        self.constraints = pk.constraint_list()

    @time_milp_generation
    def _generate_objective(
        self,
        provider_nodes: dict[str, ProviderNode] | None = None,
        station_nodes: dict[str, StationNode] | None = None,
        contact_nodes: dict[str, ContactNode] | None = None,
        satellite_nodes: dict[str, SatelliteNode] | None = None,
        opt_window: OptimizationWindow | None = None,
        **kwargs
    ):
        """
        Generate the objective function -- minimizing total gap time across all satellites.
        """

        # Group contacts by satellite
        contact_nodes_by_satellite = sorted(contact_nodes.values(), key=lambda cn: cn.satellite.id)
        self.satellite_gap_vars = dict()      # gap variables per satellite

        for sat in satellite_nodes.values():
            self.satellite_gap_vars[sat.id] = []

        # Process satellite-contact pairs for gap variables/constraints
        for sat_id, sat_contacts in groupby(contact_nodes_by_satellite, lambda cn: cn.satellite.id):
            sat_contacts = sorted(sat_contacts, key=lambda cn: cn.model.t_start)
            for i, cn_i in enumerate(sat_contacts[:-1]):
                expr = pk.expression(0)
                for cn_j in filter(lambda cn: cn.model.t_start > cn_i.model.t_end, sat_contacts):
                    pair_var = pk.variable(domain=pk.Binary)
                    self.variable_dict[(sat_id, cn_i.model.id, cn_j.model.id)] = pair_var
                    expr += pair_var

                    self.constraints.append(pk.constraint(pair_var <= contact_nodes[cn_i.id].var))
                    self.constraints.append(pk.constraint(pair_var <= contact_nodes[cn_j.id].var))

                    gap_var = pk.variable(domain=pk.NonNegativeReals)
                    self.satellite_gap_vars[sat_id].append(gap_var)

                    self.constraints.append(pk.constraint(
                        gap_var >= (cn_j.model.t_start - cn_i.model.t_end) * pair_var))
                    self.constraints.append(pk.constraint(
                        gap_var <= (cn_j.model.t_start - cn_i.model.t_end) * pair_var))

                self.constraints.append(pk.constraint(expr == contact_nodes[cn_i.id].var))

        # Set the objective to minimize the sum of all gap variables for all satellites
        all_gap_vars = []
        for gap_list in self.satellite_gap_vars.values():
            all_gap_vars.extend(gap_list)
        self.obj.expr = pk.expression(sum(all_gap_vars))
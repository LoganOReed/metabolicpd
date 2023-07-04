import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from numpy import diff

# Reads xlsx files and gives a number of plots


class Data_Reader:
    def __init__(self, filename):
        self.file_name = filename
        self.xls = pd.ExcelFile(self.file_name)
        self.sim_data = pd.read_excel(self.xls, sheet_name="sim data")
        self.time = self.sim_data.iloc[:, 0].tolist()
        print(self.time)
        self.solution = self.sim_data.iloc[:, 1:].values.tolist()
        print(self.solution)
        self.lookup_df = pd.read_excel(
            self.xls, sheet_name="lookup dict", index_col=0, header=0
        )

        self.meta_lookup = self.lookup_df.to_dict("dict")[
            0
        ]  # has the form of dict {0 : dictionary I want}

    def set_metabolites_to_plot(self, list_of_metabolites):
        self.metabolites_to_plot = list_of_metabolites

    def overlapping_plot_results_of_interest(
        self, existing_ax, label_modifier, linestyle
    ):  # , existing_ax, label_modifer):
        """Plots the results of the simulation for a subset of metabolites.

        Args:
            t: a list of timepoints
            values: a list of metabolite levels during simulation (parallel with t)
            meta_of_interest: a list of metabolite names to plot
            actual_lookup: a python dictionary where keys are metabolites, values are indices of metabolites

        Returns:
            None: displays a matplotlib.pyplot plot
        """
        indices_of_interest = []
        for meta in self.metabolites_to_plot:
            indices_of_interest.append(self.meta_lookup[meta])
        values_of_interest = []
        for entry in self.solution:
            vals = [entry[idx] for idx in indices_of_interest]
            values_of_interest.append(vals)

        if existing_ax is None:
            fig, ax = plt.subplots()
            ax.plot(
                self.time,
                values_of_interest,
                label=[label_modifier + meta for meta in self.metabolites_to_plot],
                linewidth=3,
            )
            # plt.ylim([0, 3])
            # plt.legend()
            return ax
        else:
            # old_labels = existing_ax.labels
            new_labels = [label_modifier + meta for meta in self.metabolites_to_plot]
            existing_ax.plot(
                self.time,
                values_of_interest,
                label=new_labels,
                linestyle=linestyle,
                linewidth=3,
            )
            return existing_ax

    def plot_results_of_interest(self, label_modifier):
        """Plots the results of the simulation for a subset of metabolites.

        Args:
            t: a list of timepoints
            values: a list of metabolite levels during simulation (parallel with t)
            meta_of_interest: a list of metabolite names to plot
            actual_lookup: a python dictionary where keys are metabolites, values are indices of metabolites

        Returns:
            None: displays a matplotlib.pyplot plot
        """
        indices_of_interest = []
        for meta in self.metabolites_to_plot:
            indices_of_interest.append(self.meta_lookup[meta])
        values_of_interest = []
        for entry in self.solution:
            vals = [entry[idx] for idx in indices_of_interest]
            values_of_interest.append(vals)

        fig, ax = plt.subplots()
        ax.plot(
            self.time,
            values_of_interest,
            label=[label_modifier + meta for meta in self.metabolites_to_plot],
            linewidth=3,
        )
        plt.ylim([0, 3])
        plt.legend()
        return ax

    def compute_deriv_values(self):
        idx_of_clr = self.meta_lookup["clearance_0"]
        vals_of_clr = []
        for entry in self.solution:
            val = entry[idx_of_clr]
            vals_of_clr.append(val)

        dx = diff(self.time)
        dy = diff(vals_of_clr)

        derivs = dy / dx  # pad with a leading 0
        deriv_list = [0] + list(derivs)
        self.derivs = deriv_list

    def update_clearance_values(self):
        idx_of_clr = self.meta_lookup["clearance_0"]
        for i in range(0, len(self.solution)):
            entry = self.solution[i]
            entry[idx_of_clr] = self.derivs[i]


def determine_different_trajectories(dr_low, dr_high):
    different_trajectories = []
    sig_cutoff = 0.5

    low_lookup = dr_low.meta_lookup
    high_lookup = dr_high.meta_lookup

    low_keys = low_lookup.keys()
    for key in low_keys:
        low_idx = low_lookup[key]
        high_idx = high_lookup[key]

        low_vals = [entry[low_idx] for entry in dr_low.solution]
        high_vals = [entry[high_idx] for entry in dr_high.solution]

        dif = [abs(low_vals[i] - high_vals[i]) for i in range(0, len(low_vals))]
        total = 0
        for entry in dif:
            total = total + entry

        if total > sig_cutoff:
            different_trajectories.append(key)

    return different_trajectories


def plot_difference(dr_healthy, dr_disease, title):
    healthy_sol = dr_healthy.solution
    disease_sol = dr_disease.solution

    healthy_lookup = dr_healthy.meta_lookup
    disease_lookup = dr_disease.meta_lookup

    metabolite_list = []  # to keep track of the order in which we see the keys
    trajectory_difference = []
    for key in healthy_lookup.keys():
        healthy_idx = healthy_lookup[key]
        disease_idx = disease_lookup[key]

        healthy_vals = [entry[healthy_idx] for entry in healthy_sol]
        disease_vals = [entry[disease_idx] for entry in disease_sol]

        difference = [
            disease_vals[i] - healthy_vals[i] for i in range(0, len(healthy_vals))
        ]
        if sum([abs(entry) for entry in difference]) > 0.3:
            metabolite_list.append(key)
            trajectory_difference.append(difference)

    transpose = [list(i) for i in zip(*trajectory_difference)]
    plt.plot(dr_healthy.time, transpose, label=metabolite_list)
    plt.plot(
        dr_healthy.time, [0] * len(dr_healthy.time), linestyle="dashed", color="black"
    )
    plt.legend()
    plt.ylabel("differenc in metabolite level")
    plt.xlabel("time in hours")
    plt.title(title)
    # plt.ylim([-0.75, 0.75])  # for gba
    plt.ylim([-0.2, 0.2])
    plt.show()

    return metabolite_list, trajectory_difference


if __name__ == "__main__":
    # define the file names that we want to compare
    low_gba_file_name = "data/low_gba_12h.xlsx"
    high_gba_file_name = "data/high_gba_12h.xlsx"
    low_mutant_lrrk2_file_name = "data/low_mutant_lrrk2_12h.xlsx"
    high_mutant_lrrk2_file_name = "data/high_mutant_lrrk2_12h.xlsx"

    # declare the data reader objects
    dr_low_gba = Data_Reader(low_gba_file_name)
    dr_high_gba = Data_Reader(high_gba_file_name)

    dr_low_mutant_lrrk2 = Data_Reader(low_mutant_lrrk2_file_name)
    dr_high_mutant_lrrk2 = Data_Reader(high_mutant_lrrk2_file_name)

    # which metabolites do we want to plot
    metas_to_plot = [
        "glucosylceramide_0",
        "gba_0",
        "a_syn_0",
        "a_syn_1",
        "mis_a_syn_0",
        "mis_a_syn_1",
        "a_syn_proto_0",
        "mutant_lrrk2_0",
        "clearance_0",
    ]

    # reconstruct the derivative of clearance
    dr_objs_list = [dr_low_gba, dr_high_gba, dr_low_mutant_lrrk2, dr_high_mutant_lrrk2]
    for dr_obj in dr_objs_list:
        dr_obj.compute_deriv_values()
        dr_obj.update_clearance_values()

    sig_different_traj_gba = determine_different_trajectories(dr_low_gba, dr_high_gba)
    sig_different_traj_lrrk2 = determine_different_trajectories(
        dr_low_mutant_lrrk2, dr_high_mutant_lrrk2
    )

    # update the data reader object to plot only the listed metabolites
    # uncomment the below code to reshow the plots!
    dr_low_gba.set_metabolites_to_plot(sig_different_traj_gba)
    dr_high_gba.set_metabolites_to_plot(sig_different_traj_gba)
    dr_low_mutant_lrrk2.set_metabolites_to_plot(sig_different_traj_lrrk2)
    dr_high_mutant_lrrk2.set_metabolites_to_plot(sig_different_traj_lrrk2)

    cmap = plt.get_cmap("tab20")
    gba_num_colors = len(sig_different_traj_gba)
    lrrk2_num_colors = len(sig_different_traj_lrrk2)

    gba_cycler = cycler(color=cmap.colors[0:gba_num_colors])
    lrrk2_cycler = cycler(color=cmap.colors[0:lrrk2_num_colors])
    plt.rc("axes", prop_cycle=gba_cycler)
    plt.rc("axes", titlesize=24)
    plt.rc("axes", labelsize=18)
    plt.rc("legend", fontsize=12)

    # plot the metabolites
    # ax_high_gba = dr_high_gba.overlapping_plot_results_of_interest(existing_ax=None, label_modifier='high_gba_', linestyle='solid')
    # ax_low_gba = dr_low_gba.overlapping_plot_results_of_interest(existing_ax=ax_high_gba, label_modifier='low_gba_', linestyle='dotted')
    # plt.legend()
    # plt.ylim([0, 3])
    #
    # plt.title('Low vs High GBA Levels')
    # plt.xlabel('time (unitless)')
    # plt.ylabel('metabolite level (unitless)')
    # plt.show()
    #
    # ax_low_mutant = dr_low_mutant_lrrk2.overlapping_plot_results_of_interest(existing_ax=None, label_modifier='low_mut_', linestyle='solid')
    # ax_high_mutant = dr_high_mutant_lrrk2.overlapping_plot_results_of_interest(existing_ax=ax_low_mutant, label_modifier='high_mut_', linestyle='dotted')
    # plt.rc('axes', prop_cycle=lrrk2_cycler)
    # plt.legend()
    # plt.ylim([0, 3])
    # plt.title('Low vs High Mutant LRRK2 Levels')
    # plt.xlabel('time (unitless)')
    # plt.ylabel('metabolite level (unitless)')
    # plt.show()

    plot_difference(
        dr_healthy=dr_high_gba, dr_disease=dr_low_gba, title="GBA: Disease - Healthy"
    )
    plot_difference(
        dr_healthy=dr_low_mutant_lrrk2,
        dr_disease=dr_high_mutant_lrrk2,
        title="LRRK2: Disease - Healthy",
    )

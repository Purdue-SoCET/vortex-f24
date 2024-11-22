import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    source_file = "expirements.json"
    expirements = {}

    num_scalar_idx              = 0
    theta_idx                   = 1
    avg_num_cycles_saved_idx    = 2
    avg_pct_cycles_saved_idx    = 3
    avg_speed_up_idx            = 4
    avg_rel_simd_efficiency_idx = 5
    avg_simd_efficiency_idx     = 6
    avg_pct_non_split_div_idx   = 7
    avg_max_ocp_idx             = 8
    avg_pct_max_cap_idx         = 9
    avg_num_scalarizations_idx  = 10
    avg_num_reconvergences_idx  = 11

    num_scalar_data = {}

    with open(source_file, "r") as f:
        expirements = json.load(f)

    for key in expirements.keys():
        expirement = expirements[key]

        num_scalar = expirement[num_scalar_idx]
        
        
        if num_scalar not in num_scalar_data.keys():
            num_scalar_data[num_scalar] = []
        
        num_scalar_data[num_scalar].append(expirement[theta_idx:avg_num_reconvergences_idx])
        
    for num_scalar in num_scalar_data.keys():
        data = num_scalar_data[num_scalar]

        thetas = [expr[theta_idx-1] for expr in data]
        avg_speed_ups = [expr[avg_speed_up_idx-1] for expr in data]
        avg_pct_cycles_saved = [expr[avg_pct_cycles_saved_idx-1] for expr in data]

        avg_rel_simd_efficiency = [expr[avg_rel_simd_efficiency_idx-1] for expr in data]
        avg_simd_efficiency = [expr[avg_simd_efficiency_idx-1] for expr in data]
        avg_diff_simd_efficiency = [avg_simd_efficiency[i] - avg_rel_simd_efficiency[i] for i in range(len(avg_simd_efficiency))]

        avg_pct_non_split_div = [expr[avg_pct_non_split_div_idx-1] for expr in data]
        avg_max_ocp = [expr[avg_max_ocp_idx-1] for expr in data]
        avg_pct_max_cap = [expr[avg_pct_max_cap_idx-1] for expr in data]
        avg_num_scalarizations = [expr[avg_num_scalarizations_idx-1] for expr in data]

        plt.plot(thetas, avg_speed_ups)
        plt.xlabel("Saturation Limit")
        plt.ylabel("Speed Ups (%)")
        plt.title(f'Scalarization Bandwidth: {num_scalar}')
        plt.savefig(f'plots/avg_speed_ups_{num_scalar}.png')
        plt.close()

        plt.plot(thetas, avg_pct_cycles_saved)
        plt.xlabel("Saturation Limit")
        plt.ylabel("Percentage Cycles Saved")
        plt.title(f'Scalarization Bandwidth: {num_scalar}')
        plt.savefig(f'plots/avg_pct_cycles_saved_{num_scalar}.png')
        plt.close()

        plt.plot(thetas, avg_diff_simd_efficiency)
        plt.xlabel("Saturation Limit")
        plt.ylabel("SIMD Efficiency Difference")
        plt.title(f'Scalarization Bandwidth: {num_scalar}')
        plt.savefig(f'plots/avg_diff_simd_efficiency_{num_scalar}.png')
        plt.close()

        plt.plot(thetas, avg_pct_non_split_div)
        plt.xlabel("Saturation Limit")
        plt.ylabel("Percentage of Non-Split Divergence")
        plt.title(f'Scalarization Bandwidth: {num_scalar}')
        plt.savefig(f'plots/avg_pct_non_split_div_{num_scalar}.png')
        plt.close()

        plt.plot(thetas, avg_max_ocp)
        plt.xlabel("Saturation Limit")
        plt.ylabel("Max Occupancy")
        plt.title(f'Scalarization Bandwidth: {num_scalar}')
        plt.savefig(f'plots/avg_max_ocp_{num_scalar}.png')
        plt.close()

        plt.plot(thetas, avg_pct_max_cap)
        plt.xlabel("Saturation Limit")
        plt.ylabel("Percentage of Max Capacity Reached")
        plt.title(f'Scalarization Bandwidth: {num_scalar}')
        plt.savefig(f'plots/avg_pct_max_cap_{num_scalar}.png')
        plt.close()

        plt.plot(thetas, avg_num_scalarizations)
        plt.xlabel("Saturation Limit")
        plt.ylabel("Number of Scalarizations")
        plt.title(f'Scalarization Bandwidth: {num_scalar}')
        plt.savefig(f'plots/avg_num_scalarizations_{num_scalar}.png')
        plt.close()
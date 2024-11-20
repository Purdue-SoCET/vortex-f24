##########################################################################
##########################################################################
## Purpose: To simulate runtime conditions of different divergence
## detection heuristics assuming a SIMT core architecture as the one 
## refreneced by Prof. Tim Rogers in the ECE 60827
## 
## Date: Nov 18, 2024
## Author: Hassan Al-alawi
##########################################################################
import re

def count_active_threads(tmask):
	count = 0

	for bit in tmask:
		if bit == 1:
			count += 1

	return count

def conv_str_to_list(tmask):
	return [1 if char == '1' else 0 for char in tmask]

def get_tid(tmask):
	return tmask.index(1)

def is_tid_on(tmask, tid):
	tmask = conv_str_to_list(tmask)
	return tmask[tid]

# Finds most recent tmask where the scalarized thread has been active
def find_reconv_tmask(idx, tmasks, tmask):

	reconverge_tmask = ""
	
	while True:
		prev_tmask = tmasks[idx]
		tmask_vec = conv_str_to_list(tmask)

		if(prev_tmask != tmask):
			tid = get_tid(tmask_vec)

			if(is_tid_on(prev_tmask, tid)):
				reconverge_tmask = prev_tmask
				break
			
			idx -= 1
		
		else:
			if(idx == 0):
				break
			idx -= 1

	return reconverge_tmask

def reconverge(tmask, tmask_status, reconverge_tmask):
	num_reconv = 0

	for key in tmask_status.keys():
		if(tmask_status[key] == 1):
			tmask_vector = conv_str_to_list(tmask)
			tid = get_tid(tmask_vector)

			if(reconverge_tmask[tid] == tmask):
				tmask_status[key] = 0
				num_reconv += 1
	
	return tmask_status, num_reconv

# Assummptions
# 1. Infinite scalar core resources (Not anymore)
# 2. Infinite and Transfer thread transfer bandwidth
# 3. Assume no stalls on SIMT Core end while waiting for scalar core to reconverge 

# Parameters
# 1. theta: Indicates sensitivity of the divergence detector. Sets a minimumn bound on
# number of divergent cycles that are counted for a given thread before the heuristic 
# elects to send the thread to the scalar core
#
# 2. num_threads: Number of threads per warp
#
# 3. tmasks: List of all the thread masks that are executed in one warp produced by
# baseline vortex GPU simulations
#
# 4. capacity: Is the maximumn capacity of the scalar core
#
# 5. num_scalar: Is the number of threads that the heuristic can send to the scalar core at any given cycle
#
# 6. instrs: Associated instruction for each tmask recorded, used primarily for finding reconvergence PC

# Outputs
# 1. speed_up: Total Number of baseline vortex cycles / Simulated cycles with scalarization guided by the heuristic
# 
# 2. scalarized_threads: Dictionary of thread_masks, and indirectly threads, that were scalarized with the value being the number of times that they were scalarized, meaning sent and recomverged to SIMT core
#
# 3. max_occupancy: The max number of threads that were living on the scalar core during the simulation
#
# 4. num_reconv: The number of times a scalarized thread has been sent back to the SIMT core

def sat_counters(tmasks, instrs, theta=1000, num_threads=32, capacity=32, num_scalar=1):
	# Baseline number of cycles executed on vortex GPU
	total_cycles         = len(tmasks)
	sim_cycles			 = 0

	occupancy			 = 0 # Indicates number of threads on the scalar core 
	max_occupancy 	     = 0 
	num_reconv 			 = 0
	
	speed_up	 		 = 0
	scalarized_threads 	 = {}
	sat_counters		 = [0]*num_threads
	tmask_status		 = {} # Checks if this specific thread or group of threads(tmask) is on the scalar core
	reconverge_tmask	 = [""]*num_threads
	
	for idx, tmask in enumerate(tmasks):
		if tmask not in tmask_status.keys():
			tmask_status[tmask] = 0
		
		# Check if tmask is on the scalar core or not
		if tmask_status[tmask] != 1:
		
			## If not on the scalar cores

			## Check if tmask count is == scalar_threads (T)
			
			tmask_vector = conv_str_to_list(tmask) # Turn tmask string into bit vector

			if(count_active_threads(tmask_vector) == num_scalar):
				### If so increment the coressponding thread's saturating counter 
				tid = get_tid(tmask_vector)
				sat_counters[tid] += 1

				### Check if count of the threads sat_counter reached threshold (theta)
				### If so check the capacity 

				if(sat_counters[tid] >= theta and occupancy < capacity):
					##### If we have capacity set the tmasks status to on the scalar core and reset the counter
					tmask_status[tmask] = 1
					sat_counters[tid] = 0

					occupancy += 1

					if(occupancy > max_occupancy):
						max_occupancy = occupancy

					if tmask not in scalarized_threads.keys():
						scalarized_threads[tmask] = 0

					scalarized_threads[tmask] += 1
					reconverge_tmask[tid] = find_reconv_tmask(idx-1, tmasks, tmask)



			## Check if the tmask is a reconvergence tmask for any of the threads
			reconv_threads = 0 # Refers to the number of threads that have reconverged in this cycle
			tmask_status, reconv_threads = reconverge(tmask, tmask_status, reconverge_tmask)

			occupancy -= reconv_threads
			num_reconv += reconv_threads

			## Increment sim_cycles
			sim_cycles += 1

		## If on the scalar core don't increment sim cycles as it is running in parallel with the other tmasks
		else:
			pass

	speed_up = total_cycles/sim_cycles
	cycles_saved = total_cycles - sim_cycles

	return speed_up, scalarized_threads, max_occupancy, num_reconv, cycles_saved


			

if __name__ == "__main__":
	# Get all Thread Masks from run.log
	source = "bfs_run.log"

	tmasks = []
	instrs = []
	unique_tmasks = {}
	unique_instr = {}
	num_threads = 32

	probe_warp = "0" #Specifies which warps execution stream should the parser monitor
	convergent_mask = "" 
	in_kernel = 0

	# Read from run.log
	try:
		file = open(source, 'r', errors="ignore")

		while True:
			line = file.readline()
		
			config_pattern = r"CONFIGS: num_threads=([0-9]*), num_warps=([0-9]*), num_cores=([0-9]*), num_clusters=([0-9]*), socket_size=([0-9]*), local_mem_base=(0x([a-f]|[0-9])*), num_barriers=([1-9]+)"
			tmask_pattern = r"DEBUG Fetch: cid=([0-9]+), wid=([0-9]+), tmask=([1|0]+), PC=0x(.*) \(#.*\)"
			instr_pattern = r"DEBUG Instr 0x(.+): ([A-Z]+[.]?([A-Z])*) (.*)"
			end_pattern = r"make: Leaving directory '(.+)'"

			if(re.search(config_pattern,line)):
				num_threads = int(re.search(config_pattern,line).group(1))
				for i in range(num_threads):
					convergent_mask += "1"

			if(re.search(tmask_pattern,line)):
				core_id = re.search(tmask_pattern,line).group(1)
				warp_id = re.search(tmask_pattern,line).group(2)
				tmask = re.search(tmask_pattern,line).group(3)

				instr_line = file.readline()
				instr = re.search(instr_pattern,instr_line).group(2)
				
				# Once we see a mask with all 1s, then we have entered the kernel
				if(tmask == convergent_mask and core_id == "0" and warp_id == "0"):
					in_kernel = 1

				# End of kernel and return to scheduler (Not applicable to kernels that use TMC, like BFS)
				# if(instr == "TMC"): 
				# 	in_kernel = 0
				
				if(core_id == "0" and warp_id == probe_warp and in_kernel):
					tmasks.append(tmask)
					instrs.append(instr)
					unique_tmasks[tmask] = "visited"
					unique_instr[instr] = "visited"

			if(re.search(end_pattern,line)):
				break

		file.close()

	except:
		print(line)
		print(instr_line)
		print("Error opening the log file f'{source}")


	# Call heuristic on recorded thread masks 
	thetas = [100, 500, 1000, 2000]

	print("Number of Thread Masks Processed:", len(tmasks))

	speed_up, scalarized_threads, max_occupancy, num_reconv, cycles_saved = sat_counters(tmasks, instrs, theta=1000, num_threads=num_threads, capacity=32)
	num_scalarizations = 0

	for scalar_tmask in scalarized_threads.keys():
		num_scalarizations += scalarized_threads[scalar_tmask]

	print("******************************************")
	print("Saturating Counters:")
	print(f'Number of Cycles Saved:           {cycles_saved}')
	print(f'Fraction of Total Cycles Saved:   {cycles_saved*100/len(tmasks)}')
	print(f'Speed Up: 		          {speed_up}')
	print(f'Max Occupancy:		          {max_occupancy}')
	print(f'Number of Scalarizations: 	  {num_scalarizations}')
	print(f'Number of Reconvergences:         {num_reconv}')
	print(f'Scalarised Threads:')
	print("Thread Mask           	         | Number of Times Scalarized")
	for tmask in scalarized_threads.keys():
		print(f'{tmask} | {scalarized_threads[tmask]}')
	print("******************************************")

	# for theta in thetas:
	# 	sat_counters(tmasks, theta, num_threads)
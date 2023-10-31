#include "rsbench.h"

int main(int argc, char * argv[])
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================

	int version = 13;
	double start, stop;
	
	// Process CLI Fields
	Input input = read_CLI( argc, argv );
	
	// =====================================================================
	// Print-out of Input Summary
	// =====================================================================
	logo(version);
	center_print("INPUT SUMMARY", 79);
	border_print();
	print_input_summary(input);

	// =====================================================================
	// Intialize Simulation Data Structures
	// =====================================================================
	border_print();
	center_print("INITIALIZATION", 79);
	border_print();
	
	start = get_time();
	
	SimulationData SD = initialize_simulation( input );
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
	SimulationData GSD = move_simulation_data_to_device( input, SD );
#else
	SimulationData GSD = SD;
#endif

	stop = get_time();

	printf("Initialization Complete. (%.2lf seconds)\n", stop-start);
	
	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation Begins
	// =====================================================================
	border_print();
	center_print("SIMULATION", 79);
	border_print();

	unsigned long vhash = 0;

	// Run Simulation
	start = get_time();

	// Run simulation
	if( input.simulation_method == EVENT_BASED )
	{
		run_event_based_simulation(input, GSD, &vhash );
	}
	else if( input.simulation_method == HISTORY_BASED )
	{
		printf("History-based simulation not implemented in OpenMP offload code. Instead,\nuse the event-based method with \"-m event\" argument.\n");
		exit(1);
	}

	stop = get_time();

	// Final hash step
	vhash = vhash % 999983;

	printf("Simulation Complete.\n");

	release_memory(SD);
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
	release_device_memory(GSD);
#endif


	// =====================================================================
	// Print / Save Results and Exit
	// =====================================================================
	border_print();
	center_print("RESULTS", 79);
	border_print();

	int is_invalid = validate_and_print_results(input, stop-start, vhash);

	border_print();

	return is_invalid;
}

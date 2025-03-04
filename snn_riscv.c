#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "weights.h"

// LIF neuron parameters
#define TAU_M 4.0
#define V_THRESHOLD 0.3
#define V_RESET 0.0
#define DT 1.0
#define LEAK_FACTOR 0.95
#define PAGE_SIZE 4096 // Assuming a page size of 4096 bytes

// LIF neuron model
float lif_neuron(float *voltage, float input_current) {
    *voltage = *voltage * LEAK_FACTOR + input_current; // Integrate current with leak
    if (*voltage >= V_THRESHOLD) {
        *voltage = V_RESET;
        return 1.0f; // Spike occurred
    }
    return 0.0f; // No spike
}

// Fully connected layer
void fully_connected_layer(float *input_spikes,
                           float *output_currents,
                           const int8_t *weights,
                           const int8_t *biases,
                           int output_size,
                           int input_size,
                           float weight_scale) {
    for (int i = 0; i < output_size; i++) {
        output_currents[i] = biases[i] * 2.0 / 256.0;  // Boost biases and scale
        for (int j = 0; j < input_size; j++) {
            output_currents[i] += input_spikes[j] * weights[i * input_size + j] * weight_scale / 256.0;
        }
    }
}

// SNN inference function
void snn_inference(float input_spikes[TIME_STEPS][NUM_INPUTS],
                   float output_spikes[TIME_STEPS][NUM_OUTPUTS]) {
    float voltages_hidden1[NUM_HIDDEN1] = {0};
    float voltages_hidden2[NUM_HIDDEN2] = {0};
    float voltages_output[NUM_OUTPUTS] = {0};
    
    float spikes_hidden1[NUM_HIDDEN1] = {0};
    float spikes_hidden2[NUM_HIDDEN2] = {0};

    for (int t = 0; t < TIME_STEPS; t++) {
        // Layer 1: Input to Hidden 1
        float input_current1[NUM_HIDDEN1];
        fully_connected_layer(input_spikes[t], input_current1,
                              fc1_weight, fc1_bias,
                              NUM_HIDDEN1, NUM_INPUTS, 5.0);  // 5x weight scale

        for (int i = 0; i < NUM_HIDDEN1; i++) {
            spikes_hidden1[i] = lif_neuron(&voltages_hidden1[i], input_current1[i]);
        }

        // Layer 2: Hidden 1 to Hidden 2
        float input_current2[NUM_HIDDEN2];
        fully_connected_layer(spikes_hidden1, input_current2,
                              fc2_weight, fc2_bias,
                              NUM_HIDDEN2, NUM_HIDDEN1, 3.0);  // 3x weight scale

        for (int i = 0; i < NUM_HIDDEN2; i++) {
            spikes_hidden2[i] = lif_neuron(&voltages_hidden2[i], input_current2[i]);
        }

        // Output Layer: Hidden 2 to Output
        float input_current3[NUM_OUTPUTS];
        fully_connected_layer(spikes_hidden2, input_current3,
                              fc3_weight, fc3_bias,
                              NUM_OUTPUTS, NUM_HIDDEN2, 2.0);  // 2x weight scale

        for (int i = 0; i < NUM_OUTPUTS; i++) {
            output_spikes[t][i] = lif_neuron(&voltages_output[i], input_current3[i]);
        }

        // Debugging output
        printf("t=%02d L1: V=%.2f/I=%.2f/Spk=%d | L2: V=%.2f/I=%.2f/Spk=%d | Out: V=%.2f/I=%.2f\n",
               t, 
               voltages_hidden1[0], input_current1[0], (int)spikes_hidden1[0],
               voltages_hidden2[0], input_current2[0], (int)spikes_hidden2[0],
               voltages_output[0], input_current3[0]);
    }
}

void calculate_precision(int true_positives, int false_positives) {
    float precision = (float)true_positives / (true_positives + false_positives);
    printf("Precision: %.2f%%\n", precision * 100);
}

void get_memory_usage() {
    FILE* file = fopen("/proc/self/statm", "r");
    if (!file) {
        perror("fopen");
        return;
    }
    
    long size;
    fscanf(file, "%ld", &size);
    fclose(file);

    printf("Memory Usage: %ld KB\n", size * PAGE_SIZE / 1024);
}

void get_power_consumption() {
    // This is a placeholder. Actual implementation depends on your hardware and OS.
    printf("Power Consumption: 0.00 W (placeholder value)\n");
}

// Dummy initialize_weights function (replace with actual implementation)
void initialize_weights() {
    // Your code to initialize weights
    printf("Weights initialized\n");
}

int main() {
    // Initialize weights and biases
    initialize_weights();

    // Input and output spike arrays
    float input_spikes[TIME_STEPS][NUM_INPUTS] = {0};
    float output_spikes[TIME_STEPS][NUM_OUTPUTS] = {0};

    // Generate random input spikes (80% spike probability)
    srand(time(NULL));
    for (int t = 0; t < TIME_STEPS; t++) {
        for (int i = 0; i < NUM_INPUTS; i++) {
            input_spikes[t][i] = (rand() / (float)RAND_MAX) > 0.2 ? 1.0f : 0.0f;
        }
    }

    // Run SNN inference with latency measurement
    clock_t start, end;
    double cpu_time_used;
    
    start = clock();
    snn_inference(input_spikes, output_spikes);
    end = clock();
    
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Latency: %.6f seconds\n", cpu_time_used);

    // Display memory usage
    get_memory_usage();
    
    // Placeholder for power consumption
    get_power_consumption();

    // Calculate and display precision
    int true_positives = 100;  // Placeholder value
    int false_positives = 10;  // Placeholder value
    calculate_precision(true_positives, false_positives);

    // Print final output spikes
    printf("\nFinal Spikes:\n");
    for (int t = 0; t < TIME_STEPS; t++) {
        if (t % 5 == 0) printf("t=%02d: ", t);
        for (int i = 0; i < NUM_OUTPUTS; i++) {
            if (output_spikes[t][i]) printf("%d@%d ", i, t);
        }
        if (t % 5 == 4) printf("\n");
    }

    return 0;
}

#!/bin/sh

# Directory to save the profiling results
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# Output file
OUTPUT_FILE="$OUTPUT_DIR/shell_profiling.csv"

# Environment name
ENV_NAME="test-speed"

# Number of trials
TRIALS=10

# Write CSV header
echo "trial,creation_time,deletion_time" > "$OUTPUT_FILE"

# Run trials
for TRIAL in $(seq 1 $TRIALS)
do
    echo "Starting trial $TRIAL..."

    # Profile environment creation
    START_TIME=$(date +%s)
    conda create -y -n "$ENV_NAME" python=3.9 > /dev/null 2>&1
    END_TIME=$(date +%s)
    CREATION_TIME=$((END_TIME - START_TIME))
    echo "Trial $TRIAL: Environment creation took $CREATION_TIME seconds."

    # Profile environment deletion
    START_TIME=$(date +%s)
    conda env remove -y -n "$ENV_NAME" > /dev/null 2>&1
    END_TIME=$(date +%s)
    DELETION_TIME=$((END_TIME - START_TIME))
    echo "Trial $TRIAL: Environment deletion took $DELETION_TIME seconds."

    # Append results to CSV
    echo "$TRIAL,$CREATION_TIME,$DELETION_TIME" >> "$OUTPUT_FILE"
done

echo "Profiling complete. Results saved to $OUTPUT_FILE."
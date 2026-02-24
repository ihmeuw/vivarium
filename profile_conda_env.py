import subprocess
import time
import csv
import os
import argparse

def create_conda_env(env_name):
    start_time = time.time()
    subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.9"], check=True)
    end_time = time.time()
    return end_time - start_time

def delete_conda_env(env_name):
    start_time = time.time()
    subprocess.run(["conda", "env", "remove", "-y", "-n", env_name], check=True)
    end_time = time.time()
    return end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Profile conda environment creation and deletion.")
    parser.add_argument("output_dir", type=str, help="Directory to save the profiling results.")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    env_name = "test-speed"
    results = []

    for trial in range(1, 11):
        print(f"Starting trial {trial}...")

        # Profile environment creation
        creation_time = create_conda_env(env_name)
        print(f"Trial {trial}: Environment creation took {creation_time:.2f} seconds.")

        # Profile environment deletion
        deletion_time = delete_conda_env(env_name)
        print(f"Trial {trial}: Environment deletion took {deletion_time:.2f} seconds.")

        # Record results
        results.append({
            "trial": trial,
            "creation_time": creation_time,
            "deletion_time": deletion_time
        })

    # Write results to CSV
    output_file = os.path.join(output_dir, "environment_profiling.csv")
    with open(output_file, mode="w", newline="") as csvfile:
        fieldnames = ["trial", "creation_time", "deletion_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"Profiling complete. Results saved to {output_file}.")

if __name__ == "__main__":
    main()
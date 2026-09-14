import sys
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import os

def export_tensorboard_metrics(log_dir, output_dir):
    """
    Extracts metrics from a single TensorBoard run directory and saves them as PNG graphs.

    Args:
        log_dir (str): Path to the TensorBoard run's log directory.
        output_dir (str): Path to save the generated images.

    Returns:
        bool: True if at least one metric was exported, False otherwise.
    """
    # Read TensorBoard logs
    print(f"Reading TensorBoard logs from: {log_dir}")
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    df = reader.tensors
    if df.empty:
        print(f"  No scalar data found in '{log_dir}', skipping.")
        return False

    # Keras' TensorBoard callback writes metrics as tensor summaries rather than
    # the legacy "scalar" type, so reader.tensors also carries histograms
    # (weights/biases) and a "keras" metadata tag. Keep only the true scalars.
    df = df[df["value"].apply(lambda v: isinstance(v, (int, float)))]
    if df.empty:
        print(f"  No scalar-valued tensors found in '{log_dir}', skipping.")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Group by metric tag, then plot one line per source (e.g. train/validation)
    for tag, group in df.groupby("tag"):
        plt.figure(figsize=(10, 6))
        for dir_name, sub in group.sort_values("step").groupby("dir_name"):
            plt.plot(sub["step"], sub["value"], label=dir_name, marker="o", markersize=3)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(f"Metric: {tag}")
        plt.legend()
        plt.grid(True)

        # Save plot
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(output_dir, f"{safe_tag}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Saved: {output_path}")

    return True

def export_all_runs(logs_root_dir, output_root_dir):
    """
    Walks every run subdirectory in logs_root_dir (e.g. models/logs/<timestamp>)
    and exports its metrics into a matching subfolder under output_root_dir.

    Args:
        logs_root_dir (str): Path containing one subdirectory per TensorBoard run.
        output_root_dir (str): Path under which each run's metrics are saved,
            mirrored by run name (e.g. output_root_dir/20260504-122838/*.png).
    """
    if not os.path.isdir(logs_root_dir):
        print(f"Error: Log directory '{logs_root_dir}' does not exist.")
        sys.exit(1)

    run_names = sorted(
        name for name in os.listdir(logs_root_dir)
        if os.path.isdir(os.path.join(logs_root_dir, name))
    )
    if not run_names:
        print(f"No run subdirectories found in '{logs_root_dir}'.")
        sys.exit(1)

    exported = 0
    for run_name in run_names:
        run_log_dir = os.path.join(logs_root_dir, run_name)
        run_output_dir = os.path.join(output_root_dir, run_name)
        if export_tensorboard_metrics(run_log_dir, run_output_dir):
            exported += 1

    print(f"Done. Exported metrics for {exported}/{len(run_names)} run(s).")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python read_logs.py <logs_root_dir> <output_root_dir>")
        sys.exit(1)

    logs_root_dir = sys.argv[1]
    output_root_dir = sys.argv[2]
    export_all_runs(logs_root_dir, output_root_dir)
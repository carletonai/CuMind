"""Entry for CuMind."""

from cumind.utils.config import cfg
from cumind.utils.logger import log


def main() -> None:
    """Main function for running training example."""
    from cumind.agent.runner import inference, train
    from cumind.utils.checkpoint import find_latest_checkpoint_for_env

    train()

    ckpt = find_latest_checkpoint_for_env(cfg.env.name, cfg.workspace)
    if ckpt is None:
        raise RuntimeError("No checkpoint found for environment.")
    inference(ckpt, 500)


if __name__ == "__main__":
    workspace = cfg.load("test.json")
    print(f"Workspace directory: {workspace}")

    try:
        main()
    except Exception as e:
        log.exception(f"Exception occurred: {e}")
    finally:
        log.shutdown()
        from cumind.utils.tracing import stop_trace

        stop_trace(workspace)

    # Viewing Perfetto locally (for dev):
    # After program runs, follow the link printed in the terminal to view trace in your browser.
    # Or manually upload /tmp/profile-data to https://ui.perfetto.dev

    # Viewing Perfetto remotely (TPU, SSH):
    # If running on a remote server/TPU, set up an SSH tunnel before running main.py:
    # ssh -L 9001:127.0.0.1:9001 <your-user>@<remote-host>
    # Then, open the provided Perfetto link in your local browser.
    #
    # For Google Cloud TPU:
    # gcloud compute ssh <tpu-instance-name> -- -L 9001:127.0.0.1:9001

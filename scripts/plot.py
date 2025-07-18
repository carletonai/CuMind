#!./.venv/bin/python3.12
import datetime
import os
import re

import matplotlib.pyplot as plt


def parse_log_file(filepath):
    episode_re = re.compile(r"(?P<time>\d{2}:\d{2}:\d{2} [AP]M) - INFO - Episode (?P<ep>\d+): Reward=\s*(?P<reward>[-+]?[0-9]*\.?[0-9]+), Length=\s*\d+, Loss=(?P<loss>[-+]?[0-9]*\.?[0-9]+), Memory=.*")
    policy_loss_re = re.compile(r"(?P<time>\d{2}:\d{2}:\d{2} [AP]M) - INFO - Step\s+\d+: train/policy_loss = (?P<loss>[-+]?[0-9]*\.?[0-9]+)")
    reward_loss_re = re.compile(r"(?P<time>\d{2}:\d{2}:\d{2} [AP]M) - INFO - Step\s+\d+: train/reward_loss = (?P<loss>[-+]?[0-9]*\.?[0-9]+)")
    value_loss_re = re.compile(r"(?P<time>\d{2}:\d{2}:\d{2} [AP]M) - INFO - Step\s+\d+: train/value_loss = (?P<loss>[-+]?[0-9]*\.?[0-9]+)")
    total_loss_re = re.compile(r"(?P<time>\d{2}:\d{2}:\d{2} [AP]M) - INFO - Step\s+\d+: total_loss = (?P<loss>[-+]?[0-9]*\.?[0-9]+)")

    times, rewards, loss = [], [], []
    policy_loss, policy_loss_times = [], []
    reward_loss, reward_loss_times = [], []
    value_loss, value_loss_times = [], []
    total_loss, total_loss_times = [], []

    with open(filepath, "r") as f:
        for line in f:
            m = episode_re.match(line)
            if m:
                t = m.group("time")
                times.append(t)
                rewards.append(float(m.group("reward")))
                loss.append(float(m.group("loss")))
                continue
            m = policy_loss_re.match(line)
            if m:
                policy_loss_times.append(m.group("time"))
                policy_loss.append(float(m.group("loss")))
                continue
            m = reward_loss_re.match(line)
            if m:
                reward_loss_times.append(m.group("time"))
                reward_loss.append(float(m.group("loss")))
                continue
            m = value_loss_re.match(line)
            if m:
                value_loss_times.append(m.group("time"))
                value_loss.append(float(m.group("loss")))
                continue
            m = total_loss_re.match(line)
            if m:
                total_loss_times.append(m.group("time"))
                total_loss.append(float(m.group("loss")))
                continue

    def to_seconds(tstr):
        return datetime.datetime.strptime(tstr, "%I:%M:%S %p").time()

    def time_to_sec(t):
        return t.hour * 3600 + t.minute * 60 + t.second

    episode_times = [time_to_sec(to_seconds(t)) for t in times]
    policy_loss_times = [time_to_sec(to_seconds(t)) for t in policy_loss_times]
    reward_loss_times = [time_to_sec(to_seconds(t)) for t in reward_loss_times]
    value_loss_times = [time_to_sec(to_seconds(t)) for t in value_loss_times]
    total_loss_times = [time_to_sec(to_seconds(t)) for t in total_loss_times]

    return {
        "episode_times": episode_times,
        "rewards": rewards,
        "loss": loss,
        "policy_loss_times": policy_loss_times,
        "policy_loss": policy_loss,
        "reward_loss_times": reward_loss_times,
        "reward_loss": reward_loss,
        "value_loss_times": value_loss_times,
        "value_loss": value_loss,
        "total_loss_times": total_loss_times,
        "total_loss": total_loss,
    }


def plot_and_save(x, y, label, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=label)
    plt.xlabel("Time (s since midnight)")
    plt.ylabel(ylabel)
    plt.title(f"{label} over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parse_and_plot_rl_log.py <logfile>")
        sys.exit(1)
    logfile = sys.argv[1]
    data = parse_log_file(logfile)

    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    # Reward plot
    plot_and_save(data["episode_times"], data["rewards"], "Reward", "Reward", os.path.join(outdir, "reward.png"))

    # Episode loss plot
    if data["loss"]:
        plot_and_save(data["episode_times"], data["loss"], "Episode Loss", "Loss", os.path.join(outdir, "episode_loss.png"))

    # Policy loss plot
    if data["policy_loss"]:
        plot_and_save(data["policy_loss_times"], data["policy_loss"], "Policy Loss", "Loss", os.path.join(outdir, "policy_loss.png"))

    # Reward loss plot
    if data["reward_loss"]:
        plot_and_save(data["reward_loss_times"], data["reward_loss"], "Reward Loss", "Loss", os.path.join(outdir, "reward_loss.png"))

    # Value loss plot
    if data["value_loss"]:
        plot_and_save(data["value_loss_times"], data["value_loss"], "Value Loss", "Loss", os.path.join(outdir, "value_loss.png"))

    # Total loss plot
    if data["total_loss"]:
        plot_and_save(data["total_loss_times"], data["total_loss"], "Total Loss", "Loss", os.path.join(outdir, "total_loss.png"))


if __name__ == "__main__":
    main()

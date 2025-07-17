import wandb
import tensorflow as tf
import os
import time

# Set up wandb
wandb.init(project="demo-loop-logging", name="basic-loop")

# Set up TensorBoard logging
log_dir = "logs/demo/" + time.strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Simulated training loop
for step in range(10):
    acc = step / 10
    loss = 1 - acc

    # WandB logging
    wandb.log({"accuracy": acc, "loss": loss})

    # TensorBoard logging
    with writer.as_default():
        tf.summary.scalar("accuracy", acc, step=step)
        tf.summary.scalar("loss", loss, step=step)

    print(f"Step {step} - acc: {acc:.2f}, loss: {loss:.2f}")

writer.close()
wandb.finish()

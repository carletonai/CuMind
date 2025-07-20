#for experiment tracking and visualization
import wandb
#for logging scalars to tensorboard
import tensorflow as tf
#for file and timestamp handling
import os
import time

#set up wandb project with a run name
wandb.init(project="demo-loop-logging", name="basic-loop")

#set up TensorBoard logging in logs directory
log_dir = "logs/demo/" + time.strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

#run 10 steps, with acc increasing from 0.0 to 0.9 and loss decreasing from 1.0 to 0.1
for step in range(10):
    acc = step / 10
    loss = 1 - acc

    #log to wandb
    wandb.log({"accuracy": acc, "loss": loss})

    #log to tensorboard (same metrics)
    with writer.as_default():
        tf.summary.scalar("accuracy", acc, step=step)
        tf.summary.scalar("loss", loss, step=step)

    #print to console
    print(f"Step {step} - acc: {acc:.2f}, loss: {loss:.2f}")

#close the TensorBoard writer and finish the wandb run
writer.close()
wandb.finish()

#To view WandB logs, paste in your account API key (Must have a WandB account)
#To view TensorBoard logs, run the following command in the terminal and visit localhost:
# tensorboard --logdir=logs/demo


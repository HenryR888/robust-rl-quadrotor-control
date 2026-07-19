import numpy as np
import matplotlib.pyplot as plt

for phase in ["ppo_phase3", "ppo_phase3", "ppo_phase5"]:
    d = np.load(f"logs/{phase}/evaluations.npz")
    timesteps = d["timesteps"]
    mean_reward = d["results"].mean(axis=1)
    mean_ep_len = d["ep_lengths"].mean(axis=1)

    best = np.argmax(mean_reward)
    longest = np.argmax(mean_ep_len)

    print(f"=== {phase} ===")
    print(f"EvalCallback's 'best' (highest reward): step {timesteps[best]}, "
          f"reward={mean_reward[best]:.1f}, ep_len={mean_ep_len[best]:.1f}")
    print(f"Actual longest-surviving checkpoint: step {timesteps[longest]}, "
          f"reward={mean_reward[longest]:.1f}, ep_len={mean_ep_len[longest]:.1f}")
    print()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig.suptitle(phase)
    ax1.plot(timesteps, mean_reward)
    ax1.axvline(timesteps[best], color="g", linestyle="--", label="EvalCallback 'best'")
    ax1.set_ylabel("eval mean reward")
    ax1.legend()
    ax2.plot(timesteps, mean_ep_len)
    ax2.axhline(5000, color="r", linestyle=":", label="full episode")
    ax2.axvline(timesteps[best], color="g", linestyle="--")
    ax2.set_ylabel("eval mean ep_len")
    ax2.set_xlabel("timestep")
    ax2.legend()
    plt.show()

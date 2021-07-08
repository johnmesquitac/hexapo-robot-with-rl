
import matplotlib
import plotly.express as px
from matplotlib import pyplot as plt 
import numpy as np
import seaborn as sns
import pickle
action_space = [0, 1, 2, 3]


def evaluate_training(state, steps, rewards, training_end, training_start, NUM_EPISODES,Q):

    Q = np.array(Q)
    print(Q)
    with open('pickle/'+str(state)+'.pickle', "wb") as write:
        pickle.dump(Q, write)

    all_rewards = []
    all_rewards.append(rewards)
    mean_rate = [np.mean(rewards[n-10:n]) if n > 10 else np.mean(rewards[:n])
                 for n in range(1, len(rewards))]
    elapsed_training_time = training_end-training_start
    success_rate = np.mean(rewards)
    epochs_step_rate = np.mean(steps)

    print("\nThis environment has been solved", str(success_rate), "% of times over",  str(NUM_EPISODES), "episodes within", str(elapsed_training_time),
          "seconds and an average number of timesteps per trip of", str(epochs_step_rate))
          
    plt.figure()
    plt.plot(rewards)
    plt.plot(mean_rate)
    plt.title('Gridworld Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("imgs/mean/mean"+state+".png")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(rewards, '-g', label='reward')
    ax2 = ax1.twinx()
    ax2.plot(steps, '+r', label='step')
    ax1.set_xlabel("episode")
    ax1.set_ylabel("reward")
    ax2.set_ylabel("step")
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.title("Training Progress")
    plt.savefig(
        'imgs/trainingprocess/trainingprocess'+state+'.png')

    fig = plt.figure()
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig('imgs/steps/steps'+state+'.png')

    fig = plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.savefig('imgs/rewards/rewards'+state+'.png')

    plt.figure(figsize=(12,8))
    for a_idx in action_space:
        plt.subplot(2,2,a_idx + 1)
        sns.heatmap(Q[:,a_idx].reshape((int(state),1)), cmap='hot', vmin=np.min(Q[:,:]), vmax=np.max(Q[:,:]))
        if a_idx == 0:
            direction = 'Up'
        elif a_idx ==1:
            direction = 'Left'
        elif a_idx ==2 :
            direction = 'Down'
        else:
            direction = 'Right'
        plt.title('Q-Values for Moving {}'.format(direction))
        plt.ylabel('States')
    plt.tight_layout()
    plt.savefig('imgs/heatmap/heatmap'+state+'.png')
    fig = plt.figure()
    sns.heatmap(Q[:,:], cmap='hot', vmin=np.min(Q[:,:]), vmax=np.max(Q[:,:]))
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.title('Q-Values in Training')
    plt.tight_layout()
    plt.savefig('imgs/heatmap/heatmapQ'+state+'.png')

    if state == '399':
        fig = plt.figure()
        # Create the boxplot
        plt.boxplot(all_rewards, showfliers=False)
        ymin, ymax = plt.ylim()
        plt.yticks(np.arange(ymin,ymax, step=1))
        # Save the figure
        plt.xlabel('States')
        plt.ylabel('Rewards')
        plt.title('Rewards in Training')
        plt.savefig("imgs/boxplot/boxplot.png")

# ReDER: Related Domain Experience Replay in Deep Reinforcement Learning

Artificial Intelligence has been used to train models on increasingly
complicated games, from Chess to Go to Starcraft 2. These models have
mainly been trained on a specific game, using techniques such as
adversarial search and deep reinforcement learning (RL) with Monte Carlo
Tree Search (MCTS) to optimize performance on that single board game.
While successful, these models essentially train on exorbitant amounts
of data, and must re-learn novel games from scratch.

In the context of Atari, we hypothesize that while games have different
rules, their underlying mechanics are largely similar, so training on
one game should theoretically improve performance across other games.
Specifically, using shared experience replay, we hope to employ RL to
learn a general structure of similar games. We adopt and modify a
baseline Atari Dueling DDQN implementation ([Mnih et al.
2015](https://www.nature.com/articles/nature14236.pdf), [Wang et al.
2016](https://arxiv.org/abs/1511.06581)) for a novel approach to shared
experience replay in deep reinforcement learning.

The final technical report may be found
[here](https://github.com/ringtack/RL-gina/blob/main/report/rl-gina-report.pdf).

## Installation

First, ensure `python3.7`, `pip`, and `virtualenv` are installed. In the
base directory, Run the script

    ./create_venv.sh

(If this doesn't work, manually execute the commands within the script,
and fix runtime errors along the way).

### Mac

A few additional steps may need to be executed before running. For some
reason, the OpenAI Gym `pyglet==1.5.0` dependency is outdated, and must
be replaced by `pyglet=1.5.11`. Use `pip install pyglet==1.5.11`.

The Atari Gym environment requires additional setup:

    pip install "atari[gym, accept-rom-license]"

If the ROMs are not yet installed for whatever reason, follow the
instructions [here](https://github.com/openai/atari-py#roms).

## Run

`python3 run.py -h` should provide all necessary specifications, but
it's extremely messy; sometime, I'd like to reduce redundancies and
automate more of the metrics/checkpoints/logging procedures, but here we
are for now. Here's an expanded documentation:

    usage: run.py [-h] [--env1 ENV1] [--env2 ENV2] [--model MODEL]
                  [--weight_dir WEIGHT_DIR] [--weight_name WEIGHT_NAME]
                  [--vid1_dir VID1_DIR] [--vid1_name VID1_NAME]
                  [--vid2_dir VID2_DIR] [--vid2_name VID2_NAME]
                  [--rwd1_dir RWD1_DIR] [--rwd1_name RWD1_NAME]
                  [--rwd2_dir RWD2_DIR] [--rwd2_name RWD2_NAME] [--load LOAD]
                  [--exp] [--stack] [--states STATES] [--viz1_dir VIZ1_DIR]
                  [--viz1_name VIZ1_NAME] [--viz2_dir VIZ2_DIR]
                  [--viz2_name VIZ2_NAME] [--td1_loss TD1_LOSS]
                  [--td2_loss TD2_LOSS] [--steps STEPS]

    optional arguments:
      -h, --help            show this help message and exit
      --env1 ENV1           Atari game (supported games: Pong, Cartpole,
                            SpaceInvaders, Breakout, BeamRider) (default:
                            DemonAttack)
      --env2 ENV2           Atari game (supported games: Pong, Cartpole,
                            SpaceInvaders, Breakout, BeamRider) (default:
                            SpaceInvaders)
      --model MODEL         RL model (supported models: dqn, ddqn, dddqn)
                            (default: dqn)
      --weight_dir WEIGHT_DIR
                            Set weight directory (default: ./weights)
      --weight_name WEIGHT_NAME
                            Save weight name (default: weight-12-10-11-26)
      --vid1_dir VID1_DIR   Set video directory (default: ./gifs)
      --vid1_name VID1_NAME
                            Set video name (default: vid1-12-10-11-26)
      --vid2_dir VID2_DIR   Set video directory (default: ./gifs)
      --vid2_name VID2_NAME
                            Set video name (default: vid2-12-10-11-26)
      --rwd1_dir RWD1_DIR   Set reward directory (default: ./rwds)
      --rwd1_name RWD1_NAME
                            Set reward file name (default: reward1-12-10-11-26)
      --rwd2_dir RWD2_DIR   Set reward directory (default: ./rwds)
      --rwd2_name RWD2_NAME
                            Set reward file name (default: reward2-12-10-11-26)
      --load LOAD           Load model (default: )
      --exp                 Load experience (default: False)
      --stack               Frame stack (default: False)
      --states STATES       Visualization frames (default: viz/q_states.pickle)
      --viz1_dir VIZ1_DIR   Visualization directory (default: viz)
      --viz1_name VIZ1_NAME
                            Average Q-Value information (default:
                            qvals-12-10-11-26)
      --viz2_dir VIZ2_DIR   Visualization directory (default: viz)
      --viz2_name VIZ2_NAME
                            Average Q-Value information (default:
                            qvals-12-10-11-26)
      --td1_loss TD1_LOSS   TD Loss(?) information (default: td1-loss-12-10-11-26)
      --td2_loss TD2_LOSS   TD Loss(?) information (default: td2-loss-12-10-11-26)
      --steps STEPS         Specify number of steps [TESTING ONLY] (default: 1

To run a single model:

-   Specify which Atari game you want with `--env1=<Name>`. Please
    conform to the format provided; reference the [OpenAI
    Gym](https://gym.openai.com/envs/#atari) if you're ever unsure.
-   Optionally specify a different model with `--model=<Model>`. This
    functionality is currently useless, until we expand to different
    Deep Q-Learning methods (e.g. the [C51 Distributional
    Agent](https://arxiv.org/pdf/1707.06887.pdf), any of the popular
    Policy Gradient methods (e.g. [Vanilla
    PG](https://spinningup.openai.com/en/latest/algorithms/vpg.html),
    [TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html),
    [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html),
    or
    [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)),
    or perhaps even a [Decision
    Transformer](https://arxiv.org/pdf/2106.01345.pdf)...
-   Specify a directory to store
    `{weights, rewards, videos, visualization}` directories, as well as
    possible corresponding names (although the default one, albeit
    slightly unreadable, is a decent extensible format), using
    `{weight,rwd1,vid1,viz1}_dir` and `{weight,rwd1,vid1,viz1}_name`
    respectively. Note that **you will need to create each
    (subdirectory) yourself if it does not already exist**
    Unfortunately, Python's file writers aren't that cracked. Sadge.
    -   For additional logging, specify a `td1_loss` folder to record
        loss over time.
-   Specify whether each environment should stack frames into 4
    consecutive frames, or step with a single frame. Enabling stacked
    frames is suggested for faster convergence.
-   For logging purposes (for Q value estimation), specify a pre-defined
    collection of states on which to consistently evaluate Q-values.
-   To load saved model weights, call `--load=<path/to/weights>`. **Do
    not include the extension** (e.g. `.h5`, `.pickle`)!
    -   If desirable, re-load prior experiences as well with `--exp`.
        (This should probably be integrated with weight loading)

That should be it! It's unfortunately a lot; I will need to work on
automating this process more. To run a second model, just specify each
relevant argument again using `2` instead of `1` as their suffix.

### Some Notes

Atari environment pre-processing is done almost entirely by
`atari_wrappers.py`, a utilization provided by OpenAI. Inspect the file
if one is curious.

To regenerate Q-value states (e.g. if we want a bigger size, different
stack, etc), modify `states.py` and adjust appropriate environment names
and file paths. (Yeah another thing that needs optimization rip)

Currently, although rewards and weights are stored every `SAVE_FREQ`
steps, the rewards themselves are not loadable; thus, restarting a model
will generate a new weights/rewards/Q-values list, and reset the max.
This will hopefully be remedied soon.

To view hyperparameters used, check out
[settings.py](https://github.com/ringtack/RL-gina/blob/main/src/settings.py).
Feel free to adjust.

## Results

Yay! In general, generated gif files may be inspected in `viz/gifs`. Our
best result may be seen here:
[POGGIES_MODEL](https://raw.githubusercontent.com/ringtack/RL-gina/main/src/viz/good_gifs/POGGIES_MODEL_350000.gif).

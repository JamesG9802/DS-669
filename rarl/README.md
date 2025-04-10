````md
# Running the Speaker Listener Environment

## Training the MADDPG Model

To train the **MADDPG** model on the `speaker_listener` environment, run:

```sh
python main.py -env speaker_listener
```

This will train the model and save it to:

```
models/MADDPG/MADDPG_trained_agent_simple_listener.pt
```

## Viewing the Trained Model

To view the trained model, run:

```sh
python view.py -env speaker_listener
```

This will load the model and save a visualization as a GIF at:

```
videos/simple_spread.gif
```

## Installing Dependencies

Before running the commands, ensure all dependencies are installed by running:

```sh
pip install -r requirements.txt
```
````

Then run:

```sh
pip install "agilerl==1.0.25"
```
````

There is a pygame conflict between pettingzoo and agilerl so this will solve that problem
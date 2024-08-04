import tkinter as tk
from tkinter import ttk
import time

from stable_baselines3 import PPO

from ma_dynamic_enemies import MA_PARTY_DYNAMIC_ENEMIES

# Track CPU/RAM
import threading
import psutil
from datetime import datetime

def on_ok():
    if combo_3.get() != "Only one type":
        enemies = {
            combo_2.get(): int(spinbox_1.get()), 
            combo_3.get(): int(spinbox_4.get())
                   }
        print("Two enemy types")
    else:
        enemies = {
            combo_2.get(): int(spinbox_1.get())
                   }
        print("One enemy type")

    execute_function(enemies)

def execute_function(enemies):
    model = PPO.load("ma_models\marl_heroes_vs_goblins_v01_100000_20240803-121824")
    env = MA_PARTY_DYNAMIC_ENEMIES(render_mode="human", enemies=enemies)

    episodes = 1000
    reward = 0
    all_survive = 0
    some_survive = 0
    all_die = 0
    run_away = 0

    for episode in range(1, episodes):
        print(episode)
        
        observations, _ = env.reset()
        while env.agents:
            actions = {agent: model.predict(observations[agent])[0].item() for agent in env.agents}
            observations, rewards, _, _, _ = env.step(actions)
        # print(rewards)
        reward += rewards["rogue"]
        if rewards["rogue"] > 0:
            if rewards["rogue"] == 100:
                all_survive += 1
            else:
                some_survive += 1
        elif rewards["rogue"] == -100:
            all_die += 1
        else:
            run_away += 1
        env.close()

    result = round(reward / (episodes - 1), 2)  # Example calculation
    all_survive = round(all_survive / (episodes - 1), 2)
    some_survive = round(some_survive / (episodes - 1), 2)
    all_die = round(all_die / (episodes - 1), 2)
    run_away = round(run_away / (episodes - 1), 2)

    if len(enemies) > 1:
        message = ""
        for enemy in enemies:
            message += f"{enemies[enemy]} "
            message += f"{enemy}\n"
        message += f"\nAverage reward: {result}"
        monitor_label.config(text=f"Average reward: {result}")
    else:
        for enemy in enemies:
            type_of_enemy = enemy
            number_of_enemies = enemies[enemy]
        monitor_label.config(text=f"Average reward against {number_of_enemies} {type_of_enemy}: {result}")
    monitor_label_total_success.config(text=f"All adventurer survive: {all_survive} %")
    monitor_label_success.config(text=f"All rats die: {some_survive} %")
    monitor_label_flee.config(text=f"The adventurer run away: {run_away} %")
    monitor_label_failure.config(text=f"All adventurer die: {all_die} %")

# Create the main application window
root = tk.Tk()
root.title("Start a level 1 encounter")

# Create the second label and combo box
label_2 = ttk.Label(root, text="What enemy?")
label_2.pack(pady=10)

parameters_2 = ["Giant Rat", "Goblin", "Orc", "Half-Ogre"]
combo_2 = ttk.Combobox(root, values=parameters_2)
combo_2.current(0)  # Set default selection
combo_2.pack(pady=10)

# Create the first label and combo box
label_1 = ttk.Label(root, text="How many enemies of that type?")
label_1.pack(pady=10)

spinbox_1 = tk.Spinbox(root, from_=0, to=100, increment=1)
spinbox_1.delete(0, "end")
spinbox_1.insert(0, 6)  # Set default value to 6
spinbox_1.pack(pady=10)

# Create the third label and combo box
label_3 = ttk.Label(root, text="Is there another enemy?")
label_3.pack(pady=10)

parameters_3 = ["Only one type", "Giant Rat", "Goblin", "Orc", "Half-Ogre"]
combo_3 = ttk.Combobox(root, values=parameters_3)
combo_3.current(0)  # Set default selection
combo_3.pack(pady=10)

# Create the first label and combo box
label_4 = ttk.Label(root, text="How many enemies of that type?")
label_4.pack(pady=10)

spinbox_4 = tk.Spinbox(root, from_=0, to=100, increment=1)
spinbox_4.delete(0, "end")
spinbox_4.insert(0, 6)  # Set default value to 6
spinbox_4.pack(pady=10)

# Create a monitor label to show the result of the execute_function
monitor_label = ttk.Label(root, text="Result will be displayed here")
monitor_label.pack(pady=20)

monitor_label_total_success = ttk.Label(root, text="How often all adventurer survive.")
monitor_label_success = ttk.Label(root, text="How often all rats die.")
monitor_label_flee = ttk.Label(root, text="How often the adventurer run away.")
monitor_label_failure = ttk.Label(root, text="How often all adventurer die.")

monitor_label_total_success.pack(pady=20)
monitor_label_success.pack(pady=20)
monitor_label_flee.pack(pady=20)
monitor_label_failure.pack(pady=20)

# Create an OK button
ok_button = ttk.Button(root, text="OK", command=on_ok)
ok_button.pack(pady=20)

# Run the application
root.mainloop()
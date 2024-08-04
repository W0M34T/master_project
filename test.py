possible_enemies = {
    'goblin': 'Goblin Object',  # Replace with actual objects or identifiers
    'orc': 'Orc Object'
}
enemies = {
    'goblin': 3,  # Number of goblins
    'orc': 2      # Number of orcs
}

# "goblins": [{"hp": 7, "alive": 1} for _ in range(self.number_of_goblins)],

# Generate the enemy state set
enemy_state = {
    "enemies": []
}

for enemy_type in enemies:
    for _ in range(enemies[enemy_type]):
        enemy_state["enemies"].append(possible_enemies[enemy_type])
    

print(enemy_state)
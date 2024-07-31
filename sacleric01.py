
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

import random
import numpy as np


class SA_CLERIC(Env):

    metadata = {
        "render_modes": ["human"],
        "name": "sarl_cleric_vs_goblins_v01",
    }

    def __init__(self, render_mode="human", debug_mode=False, number_of_goblins=4):

        self.render_mode = render_mode
        self.number_of_goblins = number_of_goblins
        self.debug_mode = debug_mode
        self.possible_agents = ["rogue", "fighter", "wizard", "cleric"]

        # Move, Attack lowest HP target, Attack highest HP targetm Attack random target,
        # Rest are dummy actions to be compared to ma model
        self.action_space = Discrete(9)

        # OBSERVATION Space
        # State will include heroes' and goblins' attributes (hp, alive status, zone)
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(24,),
            dtype=np.float32,
        )

        self.base_stats = {
            "rogue": {
                "hp": 9,
                "ac": 15,
                "modifier": 3,
                "weapon": "Rapier",
                "ranged_weapon": "Short Bow",
            },
            "fighter": {"hp": 13, "ac": 16, "modifier": 3, "weapon": "Greataxe"},
            "wizard": {
                "hp": 5,
                "ac": 12,
                "modifier": 3,
                "cantrip": "Firebolt",
                "spell_dc": 13,
            },
            "cleric": {
                "hp": 10,
                "ac": 15,
                "modifier": 3,
                "weapon": "Mace",
                "cantrip": "Sacred Flame",
            },
            "goblin": {"hp": 7, "ac": 13, "to_hit_bonus": 4, "dex_modifier": 2},
        }

        self.state = None
        self.max_duration = None
        self.chase_count = None
        self.heroes_alive = None
        self.goblins_alive = None

        if self.debug_mode:
            print("Roll initiative!")

    def step(self, action):
        reward = 0
        terminations = False

        # Initiative: Rogue, Goblin, Goblin, Goblin, Goblin, Fighter, Wizard, Cleric
        # TODO: randomize initiative
        self.rogue_turn()
        self.fighter_turn()
        self.enemy_turns()
        self.wizard_turn()
        self.cleric_turn(action)

        self.max_duration -= 1

        # Calculate reward and check if fight is over
        if self.heroes_alive == 0:
            reward = -100
            terminations = True
        elif self.goblins_alive == 0:
            terminations = True
            if self.heroes_alive == 4:
                reward = 100
            else:
                reward += self.heroes_alive * 20
        elif self.max_duration <= 0 or self.chase_count >= 10:
            terminations = True
            reward = -50

        observations = {
            agent: np.concatenate(
                [
                    np.array(
                        [
                            self.state[hero]["hp"] / self.base_stats[hero]["hp"],
                            self.state[hero]["alive"],
                            self.state[hero]["zone"] / 2.0,
                            self.state[hero]["spellslots"] / 2.0,
                        ],
                        dtype=np.float32,
                    )
                    for hero in self.possible_agents
                ]
                + [
                    np.array([goblin["hp"] / 7.0, goblin["alive"]], dtype=np.float32)
                    for goblin in self.state["goblins"]
                ],
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        cleric_action_mask = np.ones(9, dtype=np.int8)
        if self.state["cleric"]["alive"] == 1:
            cleric_action_mask[8] = 0
            if self.state["cleric"]["spellslots"] <= 0:
                cleric_action_mask[1] = (
                    0  # Action 1 is Cast Spell Action (Healing Word, Cure Wounds)
                )
        else:
            cleric_action_mask[0:8] = 0

        info = {"action_mask": cleric_action_mask}


        # Return step information
        return observations, reward, terminations, None, info

    def render(self):
        pass

    def reset(self,  seed=None, options=None):
        self.state = {
            "rogue": {"hp": 9, "alive": 1, "zone": 2, "spellslots": 0},
            "fighter": {"hp": 13, "alive": 1, "zone": 2, "spellslots": 0},
            "wizard": {"hp": 5, "alive": 1, "zone": 1, "spellslots": 2},
            "cleric": {"hp": 10, "alive": 1, "zone": 1, "spellslots": 2},
            "goblins": [{"hp": 7, "alive": 1} for _ in range(self.number_of_goblins)],
        }

        self.max_duration = 1000
        self.heroes_alive = 4
        self.goblins_alive = 4

        self.goblins_alive = self.number_of_goblins

        observations = {
            agent: np.concatenate(
                [
                    np.array(
                        [
                            self.state[hero]["hp"] / self.base_stats[hero]["hp"],
                            self.state[hero]["alive"],
                            self.state[hero]["zone"] / 2.0,
                            self.state[hero]["spellslots"] / 2.0,
                        ],
                        dtype=np.float32,
                    )
                    for hero in self.possible_agents
                ]
                + [
                    np.array([goblin["hp"] / 7.0, goblin["alive"]], dtype=np.float32)
                    for goblin in self.state["goblins"]
                ],
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        cleric_action_mask = np.ones(9, dtype=np.int8)
        if self.state["cleric"]["alive"] == 1:
            cleric_action_mask[8] = 0
            if self.state["cleric"]["spellslots"] <= 0:
                cleric_action_mask[1] = (
                    0  # Action 1 is Cast Spell Action (Healing Word, Cure Wounds)
                )
        else:
            cleric_action_mask[0:8] = 0

        info = {"action_mask": cleric_action_mask}

        return observations, info  # self.state

    
    ######################## Helper functions #############################
    
    def choose_target(self, mode):
        goblins = self.state["goblins"]

        if mode == "low":
            min_hp = min(goblin["hp"] for goblin in goblins if goblin["alive"])
            for goblin in goblins:
                if goblin["hp"] == min_hp:
                    return goblin
        elif mode == "high":
            max_hp = max(goblin["hp"] for goblin in goblins if goblin["alive"])
            for goblin in goblins:
                if goblin["hp"] == max_hp:
                    return goblin
        elif mode == "random":
            while True:
                goblin = random.choice(goblins)
                if goblin["alive"]:
                    return goblin

    def deal_damage(self, target, damage):
        if target["hp"] <= damage:
            target["hp"] = 0
            target["alive"] = 0
            self.goblins_alive -= 1
            if self.debug_mode:
                print(f"... hits for {damage} damage killing the goblin!")
        else:
            target["hp"] -= damage
            hp = target["hp"]
            if self.debug_mode:
                print(f"... hits for {damage} damage leaving the goblin at {hp} hp!")

    def attack_roll(self, to_hit_bonus, armor_class):
        return random.randint(1, 20) + to_hit_bonus >= armor_class

    def damage_roll(self, max_damage, modifier):
        return random.randint(1, max_damage) + modifier

    def saving_throw(self, saving_throw_modifier, dc):
        return random.randint(1, 20) + saving_throw_modifier >= dc
    
    ######################## Enemy Turns ##################################

    def enemy_turns(self):
        if self.goblins_alive > 0 and self.heroes_alive > 0:
            if (
                (self.state["rogue"]["zone"] != 2 or self.state["rogue"]["alive"] == 0)
                and (
                    self.state["fighter"]["zone"] != 2
                    or self.state["fighter"]["alive"] == 0
                )
                and (
                    self.state["wizard"]["zone"] != 2
                    or self.state["wizard"]["alive"] == 0
                )
                and (
                    self.state["cleric"]["zone"] != 2
                    or self.state["cleric"]["alive"] == 0
                )
            ):
                self.goblins_move()

            else:
                self.enemy_attack_action(self.goblins_alive)

    def enemy_attack_action(self, num_attacks):
        to_hit_bonus = self.base_stats["goblin"]["to_hit_bonus"]
        dex_mod = self.base_stats["goblin"]["dex_modifier"]

        for _ in range(num_attacks):
            if self.debug_mode:
                print("Goblin turn starts ...")
            if self.heroes_alive > 0 and (
                (self.state["rogue"]["zone"] == 2 and self.state["rogue"]["alive"] == 1)
                or (
                    self.state["fighter"]["zone"] == 2
                    and self.state["fighter"]["alive"] == 1
                )
                or (
                    self.state["wizard"]["zone"] == 2
                    and self.state["wizard"]["alive"] == 1
                )
                or (
                    self.state["cleric"]["zone"] == 2
                    and self.state["cleric"]["alive"] == 1
                )
            ):
                target, target_ac, target_name = self.choose_hero_target()
                if self.debug_mode:
                    print(f"... attacking {target_name} ...")
                if self.attack_roll(to_hit_bonus, target_ac):
                    damage = self.damage_roll(max_damage=6, modifier=dex_mod)
                    self.deal_damage_to_hero(target, damage, target_name)
                else:
                    if self.debug_mode:
                        print(f"... misses!")
            else:
                self.goblins_move()

    def goblins_move(self):
        self.state["rogue"]["zone"] = 2
        self.state["fighter"]["zone"] = 2
        self.state["wizard"]["zone"] = 2
        self.state["cleric"]["zone"] = 2

        if self.debug_mode:
            print("The Goblins move towards the heroes!")

    def deal_damage_to_hero(self, target, damage, target_name):
        if target["hp"] <= damage:
            target["hp"] = 0
            target["alive"] = 0
            self.heroes_alive -= 1
            if self.debug_mode:
                print(
                    f"... hits for {damage} damage hitting the {target_name} unconcious!"
                )
        else:
            target["hp"] -= damage
            hp = target["hp"]
            if self.debug_mode:
                print(
                    f"... hits for {damage} damage leaving the {target_name} at {hp} hp!"
                )

    def choose_hero_target(self, ranged_allowed=False):
        for _ in range(1000):
            selected_option = random.randint(1, 4)
            if (
                selected_option == 1
                and self.state["fighter"]["alive"]
                and (self.state["fighter"]["zone"] == 2)
            ):
                return (
                    self.state["fighter"],
                    self.base_stats["fighter"]["ac"],
                    "fighter",
                )
            elif (
                selected_option == 2
                and self.state["rogue"]["alive"]
                and (self.state["rogue"]["zone"] == 2)
            ):
                return self.state["rogue"], self.base_stats["rogue"]["ac"], "rogue"
            elif (
                selected_option == 3
                and self.state["cleric"]["alive"]
                and (self.state["cleric"]["zone"] == 2)
            ):
                return self.state["cleric"], self.base_stats["cleric"]["ac"], "cleric"
            elif (
                selected_option == 4
                and self.state["wizard"]["alive"]
                and (self.state["wizard"]["zone"] == 2)
            ):
                return self.state["wizard"], self.base_stats["wizard"]["ac"], "wizard"

    ######################## Hero Turns ###################################

    def rogue_turn(self):
        if self.debug_mode:
                print(f"Rogue turn starts ...")
                
        # Move Action (0), Attack Action (1, 2, 3)
        if self.state["rogue"]["alive"] and self.goblins_alive >= 1:
            
            # 0
            if self.state["rogue"]["hp"] <= 5 and self.state["rogue"]["zone"] != 1:
                self.move_action(self.state["rogue"])
            elif self.state["rogue"]["hp"] > 5 and self.state["rogue"]["zone"] != 2:
                self.move_action(self.state["rogue"])
                
            # 1, 2, 3
            else:
                action = random.randint(1, 3)
                if action == 1:
                    mode = "low"
                elif action == 2:
                    mode = "high"
                elif action == 3:
                    mode = "random"

                target = self.choose_target(mode)
                modifier = self.base_stats["rogue"]["modifier"]
                sneak_attack = False

                if self.state["rogue"]["zone"] == 2:
                    if (
                        self.state["fighter"]["zone"] == 2
                        or self.state["wizard"]["zone"] == 2
                        or self.state["cleric"]["zone"] == 2
                    ):
                        sneak_attack = True
                    weapon = self.base_stats["rogue"]["weapon"]
                else:
                    weapon = self.base_stats["rogue"]["ranged_weapon"]

                self.attack_action(target, modifier, weapon, sneak_attack)
        else:
            if self.debug_mode:
                print(f"... doing nothing")

    def fighter_turn(self):
        # Move Action (0), Attack Action (1, 2, 3)
        if self.debug_mode:
                print(f"Fighter turn starts ...")
                
        if self.state["fighter"]["alive"] and self.goblins_alive >= 1:
            # 0
            if self.state["rogue"]["hp"] <= 5 and self.state["rogue"]["zone"] != 1:
                self.move_action(self.state["rogue"])
            elif self.state["rogue"]["hp"] > 5 and self.state["rogue"]["zone"] != 2:
                self.move_action(self.state["rogue"])
                
            # 1, 2, 3
            else:
                action = random.randint(1, 3)
                if action == 1:
                    mode = "low"
                elif action == 2:
                    mode = "high"
                elif action == 3:
                    mode = "random"

                target = self.choose_target(mode)
                modifier = self.base_stats["fighter"]["modifier"]
                weapon = self.base_stats["fighter"]["weapon"]

                self.attack_action(target, modifier, weapon)
        else:
            if self.debug_mode:
                print(f"... doing nothing")

    def wizard_turn(self):
        if self.debug_mode:
                print(f"Wizard turn starts ...")
                
        # Move Action (0), Spell Cast Action (1), Cantrip Action (2, 3, 4)
        if self.state["wizard"]["alive"] and self.goblins_alive >= 1:
            
            # 0
            if self.state["wizard"]["hp"] <= 5 and self.state["wizard"]["zone"] != 2 and random.randint(1, 2) == 1:
                self.move_action(self.state["wizard"])
            elif self.state["wizard"]["hp"] > 5 and self.state["wizard"]["zone"] != 1:
                self.move_action(self.state["wizard"])
            
            # 1
            elif self.state["wizard"]["spellslots"] > 0 and self.state["wizard"]["zone"] != 1:
                self.cast_burning_hands_action()
                self.state["wizard"]["spellslots"] -= 1
            
            # 2, 3, 4
            else:
                action = random.randint(2, 4)
                if action == 2:
                    mode = "low"
                elif action == 3:
                    mode = "high"
                elif action == 4:
                    mode = "random"

                target = self.choose_target(mode)
                modifier = self.base_stats["wizard"]["modifier"]
                cantrip = self.base_stats["wizard"]["cantrip"]

                self.cantrip_action(target, modifier, cantrip)
        else:
            if self.debug_mode:
                print(f"... doing nothing")

    def cleric_turn(self, action):
        # Move Action (0), Spell Cast Action (1, 2, 3, 4), Cantrip Action (5, 6, 7)
        if self.state["cleric"]["alive"] and self.goblins_alive >= 1:
            if self.debug_mode:
                print(f"({action}) Cleric turn starts ...")
            if action == 0:
                self.move_action(self.state["cleric"])
            elif action in [1, 2, 3, 4] and (self.state["cleric"]["spellslots"] > 0):
                if action == 1:
                    target = self.state["fighter"]
                    target_name = "fighter"
                elif action == 2:
                    target = self.state["rogue"]
                    target_name = "rogue"
                elif action == 3:
                    target = self.state["wizard"]
                    target_name = "wizard"
                elif action == 4:
                    target = self.state["cleric"]
                    target_name = "cleric"

                if target["zone"] == self.state["cleric"]["zone"]:
                    spell = "Cure Wounds"
                else:
                    spell = "Healing Word"

                modifier = self.base_stats["cleric"]["modifier"]

                self.state["cleric"]["spellslots"] -= 1
                self.cast_heal_action(target, target_name, modifier, spell)

            elif action in [5, 6, 7]:
                if action == 5:
                    mode = "low"
                elif action == 6:
                    mode = "high"
                elif action == 7:
                    mode = "random"
                target = self.choose_target(mode)
                modifier = self.base_stats["cleric"]["modifier"]

                if self.state["cleric"]["zone"] == 1:
                    cantrip = self.base_stats["cleric"]["cantrip"]
                    self.cantrip_action(target, modifier, cantrip)

                else:
                    weapon = self.base_stats["cleric"]["weapon"]
                    self.attack_action(target, modifier, weapon)
            else:
                if self.debug_mode:
                    print(f"({action}) ... doing nothing")

    ######################## Hero Actions #################################

    def attack_action(self, target, modifier, weapon, sneak_attack=False):
        if self.debug_mode:
            print(f"... attacks a goblin with a {weapon} ...")
        to_hit_bonus = 2 + modifier

        max_weapon_damage = {
            "Mace": 8,
            "Greataxe": 12,
            "Short Sword": 6,
            "Rapier": 8,
            "Short Bow": 6,
        }
        max_damage = max_weapon_damage.get(weapon)

        hit = self.attack_roll(to_hit_bonus, self.base_stats["goblin"]["ac"])
        if hit:
            damage = self.damage_roll(max_damage, modifier)
            if sneak_attack:
                damage += self.damage_roll(6, 0)

            self.deal_damage(target, damage)
        else:
            if self.debug_mode:
                print("but misses!")

    def cantrip_action(self, target, modifier, cantrip):
        if self.debug_mode:
            print(f"... casts {cantrip} ...")
        to_hit_bonus = 2 + modifier

        max_cantrip_damage = {"Firebolt": 10, "Sacred Flame": 8}
        max_damage = max_cantrip_damage.get(cantrip)

        hit = self.attack_roll(to_hit_bonus, self.base_stats["goblin"]["ac"])
        if hit:
            damage = self.damage_roll(max_damage, 0)
            self.deal_damage(target, damage)
        else:
            if self.debug_mode:
                print("... but misses!")

    def cast_burning_hands_action(self):
        if self.debug_mode:
            print("... casts burning hands ...")
        damage = random.randint(1, 6) + random.randint(1, 6) + random.randint(1, 6)
        goblin_dex_save_mod = self.base_stats["goblin"]["dex_modifier"]
        wizard_spell_dc = self.base_stats["wizard"]["spell_dc"]

        # Assumption that two Goblins are within reach (if alive)
        if self.goblins_alive > 1:
            targets = self.choose_targets()

            for goblin in targets:
                saved = self.saving_throw(goblin_dex_save_mod, wizard_spell_dc)
                if saved:
                    damage = round(damage / 2)

                self.deal_damage(goblin, damage)
        else:
            target = self.choose_target(mode="random")

            saved = self.saving_throw(goblin_dex_save_mod, wizard_spell_dc)
            if saved:
                damage = round(damage / 2)

            self.deal_damage(target, damage)

    def choose_targets(self):
        while True:
            target1, target2 = random.sample(self.state["goblins"], 2)
            if target1["alive"] and target2["alive"]:
                return [target1, target2]

    def cast_heal_action(self, target, target_name, modifier, spell):
        if self.debug_mode:
            print(f"... casts {spell} ...")
        max_spell_heal = {"Healing Word": 4, "Cure Wounds": 8}
        disciple_of_life_bonus = self.base_stats["cleric"]["modifier"]
        max_heal = max_spell_heal.get(spell)

        heal = random.randint(1, max_heal) + modifier + disciple_of_life_bonus

        if target["alive"] == 1:
            target["hp"] += heal
            if target["hp"] > self.base_stats[target_name]["hp"]:
                target["hp"] = self.base_stats[target_name]["hp"]
                if self.debug_mode:
                    hp = target["hp"]
                    print(f"... healing {target_name} full to {hp} hp!")
            else:
                if self.debug_mode:
                    hp = target["hp"]
                    print(
                        f"... healing {target_name} for {heal} leaving him on {hp} hp!"
                    )
        else:
            target["hp"] = heal
            target["alive"] = 1
            self.heroes_alive += 1
            if self.debug_mode:
                hp = target["hp"]
                print(
                    f"... healing {target_name} for {heal} letting him walk again with {hp} hp!"
                )

    def move_action(self, character):
        if character["zone"] == 2:
            character["zone"] = 1
            if self.debug_mode:
                print("... moves to the backline!")

        elif character["zone"] == 1:
            character["zone"] = 2
            if self.debug_mode:
                print("... moves to the frontline!")
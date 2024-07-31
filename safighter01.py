from gym import Env, spaces
from gym.spaces import Discrete
import random


class SA_FIGHTER(Env):
    def __init__(self):
        self.action_space = Discrete(2)

        # OBSERVATION Space
        # actions raum erweitern auf max hp goblin und zufälliger goblin
        self.observation_space = Discrete(2)

        # 1 Alive
        # 0 Dead
        self.state = 1

        # Set Maximal Length
        self.max_duration = None

        # "Agents"
        self.heroes_alive = None
        self.fighter = None
        self.wizard = None
        self.rogue = None
        self.cleric = None

        # NPCs
        self.goblins_alive = None
        self.goblin1 = None
        self.goblin2 = None
        self.goblin3 = None
        self.goblin4 = None

    def step(self, action):
        reward = 0
        done = False

        # Initiative: Rogue, Goblin, Goblin, Goblin, Goblin, Fighter, Wizard, Cleric
        # TODO: randomize initiative
        self.rogue_turn()
        self.fighter_turn(action)
        self.goblins_actions()
        self.wizard_turn()
        self.cleric_turn()

        self.max_duration -= 1

        # Calculate reward and check if fight is over
        if self.heroes_alive == 0:
            reward = -100
            done = True
        elif self.goblins_alive == 0:
            done = True
            if self.heroes_alive == 4:
                reward = 100
            else:
                reward += self.heroes_alive * 20
        elif self.max_duration <= 0:
            done = True
            reward = -50

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = 1
        self.max_duration = 1000
        self.heroes_alive = 4
        self.fighter = {
            "hp": 13,
            "ac": 16,
            "alive": True,
            "actions": {0, 1},
            "zone": 2,
        }
        self.wizard = {
            "hp": 5,
            "ac": 12,
            "alive": True,
            "actions": {0, 1, 2},
            "spellslots": 2,
            "zone": 1,
        }
        self.rogue = {
            "hp": 9,
            "ac": 15,
            "alive": True,
            "actions": {0, 1},
            "zone": 2,
        }
        self.cleric = {
            "hp": 10,
            "ac": 15,
            "alive": True,
            "actions": {0, 1, 2, 3},
            "spellslots": 2,
            "zone": 1,
        }
        self.goblins_alive = 4
        self.ranged_goblins_alive = 2
        self.goblin1 = {"hp": 7, "ac": 13, "hit": 4, "alive": True}
        self.goblin2 = {"hp": 7, "ac": 13, "hit": 4, "alive": True}
        self.goblin3 = {"hp": 7, "ac": 13, "hit": 4, "alive": True}
        self.goblin4 = {"hp": 7, "ac": 13, "hit": 4, "alive": True}

        return self.state

    ##################################################################################

    def goblins_actions(self):
        # Move (two) / Attack Ranged (two)
        # Attack Melee (all)

        if (
            self.rogue["zone"] != 2
            and self.fighter["zone"] != 2
            and self.wizard["zone"] != 2
            and self.cleric["zone"] != 2
        ):
            # Move towards Heroes
            self.rogue["zone"] = 2
            self.fighter["zone"] = 2
            self.wizard["zone"] = 2
            self.cleric["zone"] = 2

            # Ranged Attacks
            if self.ranged_goblins_alive > 0:
                self.goblin_attack(self.ranged_goblins_alive)

        else:
            # Melee Attacks
            if self.goblins_alive > 0:
                self.goblin_attack(self.goblins_alive)

    def goblin_attack(self, i):
        ranged_attacks = self.ranged_goblins_alive

        for _ in range(i):
            if ranged_attacks > 0:
                target = self.choose_hero_target(ranged_allowed=True)
                ranged_attacks -= 1
            else:
                target = self.choose_hero_target()

            to_hit_bonus = 4
            dex_mod = 2

            # Attack Roll
            attack_roll = random.uniform(1, 20) + to_hit_bonus

            # Attack Damage
            if attack_roll >= target["ac"]:
                damage = random.uniform(1, 6) + dex_mod

                self.deal_damage_to_hero(target, damage)

    def deal_damage_to_hero(self, target, damage):
        if target["hp"] <= damage:
            target["hp"] = 0
            target["alive"] = False
            self.heroes_alive -= 1
        else:
            target["hp"] -= damage

    def choose_hero_target(self, ranged_allowed=False):
        for _ in range(1000):
            selected_option = random.uniform(1, 4)
            if (
                selected_option == 1
                and self.fighter["alive"]
                and (self.fighter["zone"] == 2 or ranged_allowed)
            ):
                return self.fighter
            elif (
                selected_option == 2
                and self.rogue["alive"]
                and (self.rogue["zone"] == 2 or ranged_allowed)
            ):
                return self.rogue
            elif (
                selected_option == 3
                and self.cleric["alive"]
                and (self.cleric["zone"] == 2 or ranged_allowed)
            ):
                return self.cleric
            elif (
                selected_option == 4
                and self.wizard["alive"]
                and (self.wizard["zone"] == 2 or ranged_allowed)
            ):
                return self.wizard
        return self.fighter

        """if selected_option == 1:
            target = self.fighter
        else:
            target = self.rogue

        return target"""

    #################################################################################

    def rogue_attack_ranged(self, target, hide):
        # Average Damage: ((21 - AC + Proficiency + Modifier) / 20) * (Schaden +  0,05 * WürfelSchaden)
        to_hit_bonus = 5
        dex_mod = 3

        # Hide
        if (random.uniform(1, 20) + 5) > 9 and hide:
            # Attack Roll Advantage
            attack_roll = (
                max(random.uniform(1, 20), random.uniform(1, 20)) + to_hit_bonus
            )
            if attack_roll >= target["ac"]:
                # Damage including Sneak Attack
                damage = random.uniform(1, 6) + random.uniform(1, 6) + dex_mod
                self.deal_damage(target, damage)
        else:
            # Attack Roll
            attack_roll = random.uniform(1, 20) + to_hit_bonus
            if attack_roll >= target["ac"]:
                # Damage excluding Sneak Attack
                damage = random.uniform(1, 6) + dex_mod
                self.deal_damage(target, damage)

    def deal_damage(self, target, damage):
        if target["hp"] <= damage:
            target["hp"] = 0
            target["alive"] = False
            self.goblins_alive -= 1
        else:
            target["hp"] -= damage

    def rogue_attack_melee(self, target):
        to_hit_bonus = 5
        dex_mod = 3

        # Attack Roll
        attack_roll = random.uniform(1, 20) + to_hit_bonus

        # Attack Damage
        if attack_roll >= target["ac"]:
            damage = random.uniform(1, 6) + dex_mod

            # Sneak Attack
            if (
                self.fighter["zone"] != 1
                or self.wizard["zone"] != 1
                or self.cleric["zone"] != 1
            ):
                damage += random.uniform(1, 6)

            self.deal_damage(target, damage)

    def fighter_attack_melee(self, target):
        to_hit_bonus = 5
        str_mod = 3

        # Attack Roll
        attack_roll = random.uniform(1, 20) + to_hit_bonus

        # Attack Damage
        if attack_roll >= target["ac"]:
            damage = random.uniform(1, 6) + random.uniform(1, 6) + str_mod
            self.deal_damage(target, damage)

    def cleric_attack_melee(self, target):
        to_hit_bonus = 5
        str_mod = 3

        # Attack Roll
        attack_roll = random.uniform(1, 20) + to_hit_bonus

        # Attack Damage
        if attack_roll >= target["ac"]:
            damage = random.uniform(1, 6) + str_mod

            self.deal_damage(target, damage)

    def heal_melee(self, target):
        spell_mod = 5
        heal = random.uniform(1, 8) + spell_mod
        target["hp"] += heal
        if target["alive"] == False:
            target["alive"] == True

    def heal_ranged(self, target):
        spell_mod = 5
        heal = random.uniform(1, 4) + spell_mod
        target["hp"] += heal
        if target["alive"] == False:
            target["alive"] == True

    def cantrip(self, target):
        spell_mod = 5

        # Attack Roll
        attack_roll = random.uniform(1, 20) + spell_mod

        # Attack Damage
        if attack_roll >= target["ac"]:
            damage = random.uniform(1, 10)
            self.deal_damage(target, damage)

    def burning_hands(self):
        # 2 targets one zone away
        dex_save = random.uniform(1, 20) + 2
        damage = random.uniform(1, 6) + random.uniform(1, 6) + random.uniform(1, 6)

        if dex_save > 13:
            damage = damage / 2

        target = self.choose_target()
        self.deal_damage(target, damage)

        target2 = self.choose_target2(target)
        self.deal_damage(target2, damage)

    def move(self, target):
        if target["zone"] == 1:
            target["zone"] == 2
        else:
            target["zone"] == 1

    def choose_target(self):
        # return goblin with lowest hp
        target = None
        if self.goblin1["alive"]:
            target = self.goblin1
        if self.goblin2["alive"]:
            if target == None:
                target = self.goblin2
            else:
                target = self.compare_hp(target, self.goblin2)
        if self.goblin3["alive"]:
            if target == None:
                target = self.goblin3
            else:
                target = self.compare_hp(target, self.goblin3)
        if self.goblin4["alive"]:
            if target == None:
                target = self.goblin4
            else:
                target = self.compare_hp(target, self.goblin4)
        return target

    def choose_random_target(self):
        for _ in range(1000):
            selected_option = random.uniform(1, 4)
            if selected_option == 1 and self.goblin1["alive"]:
                target = self.goblin1
                break
            elif selected_option == 2 and self.goblin2["alive"]:
                target = self.goblin2
                break
            elif selected_option == 3 and self.goblin3["alive"]:
                target = self.goblin3
                break
            elif selected_option == 4 and self.goblin4["alive"]:
                target = self.goblin4
                break
        return target

    def target_max_hp(self):
        # return goblin with highest hp
        target = None
        if self.goblin1["alive"]:
            target = self.goblin1
        if self.goblin2["alive"]:
            if target == None:
                target = self.goblin2
            else:
                target = self.compare_hp(target, self.goblin2, True)
        if self.goblin3["alive"]:
            if target == None:
                target = self.goblin3
            else:
                target = self.compare_hp(target, self.goblin3, True)
        if self.goblin4["alive"]:
            if target == None:
                target = self.goblin4
            else:
                target = self.compare_hp(target, self.goblin4, True)
        return target

    def choose_target2(self, nontarget):
        # return goblin with lowest hp except nontarget
        target = None
        if self.goblin1["alive"] and self.goblin1 != nontarget:
            target = self.goblin1
        if self.goblin2["alive"] and self.goblin2 != nontarget:
            if target == None:
                target = self.goblin2
            else:
                target = self.compare_hp(target, self.goblin2)
        if self.goblin3["alive"] and self.goblin3 != nontarget:
            if target == None:
                target = self.goblin3
            else:
                target = self.compare_hp(target, self.goblin3)
        if self.goblin4["alive"] and self.goblin4 != nontarget:
            if target == None:
                target = self.goblin4
            else:
                target = self.compare_hp(target, self.goblin4)
        return target

    def compare_hp(self, target, alternative, max=False):
        if max:
            if alternative["hp"] > target["hp"]:
                target = alternative
        else:
            if alternative["hp"] < target["hp"]:
                target = alternative

        return target

    ##################################################################################
    def rogue_turn(self):
        # 0 Attack + Move
        # 1 Move + Attack
        # 2 Hide + Attack
        # 3 Attack

        # Only Act if alive
        if self.rogue["alive"] and self.goblins_alive >= 1:
            # Choose target for the attack
            target = self.choose_target()

            if self.rogue["zone"] == 2:
                # Melee Attack
                self.rogue_attack_melee(target)

                # Move
                if self.rogue["hp"] < 5 and self.rogue["zone"] != 1:
                    self.rogue["zone"] = 1

            elif self.rogue["hp"] >= 5:
                # Move
                self.rogue["zone"] = 1

                # Ranged Attack
                self.rogue_attack_melee(target, hide=False)

            else:
                # Ranged Attack
                self.rogue_attack_ranged(target, hide=True)

    def fighter_turn(self, action):
        # 0 Move (from zone 1 to zone 2 or from zone 2 to zone 1)
        # 1 Attack lowest hp enemy (only from zone 1)
        # 2 Attack highest hp enemy
        # 3 Attack random hp enemy

        # Only Act if alive
        if self.fighter["alive"] and self.goblins_alive >= 1:

            # Move
            if action == 0:
                if self.fighter["zone"] == 1:
                    self.fighter["zone"] = 2
                elif self.fighter["zone"] == 2:
                    self.fighter["zone"] = 1
            # Attack
            if self.fighter["zone"] == 1 and action != 0:
                if action == 1:
                    target = self.choose_target()
                elif action == 2:
                    target = self.target_max_hp()
                elif action == 3:
                    target = self.choose_random_target()

                self.fighter_attack_melee(target)

        """
        if self.fighter["hp"] < 5 and self.fighter["zone"] != 1:
            self.fighter["zone"] = 1
        else:
            # Attack
            target = self.choose_target(self)
            self.attack(target)
        """

    def wizard_turn(self):
        # 0 Move
        # 1 Attack (Cantrip)
        # 2 Attack (Burning Hands)

        # Only Act if alive
        if self.wizard["alive"] and self.goblins_alive >= 1:
            # Move
            if self.wizard["hp"] < 5 and self.wizard["zone"] != 1:
                self.wizard["zone"] = 1
            # Attack (Burning Hands)
            elif self.wizard["spellslots"] < 0:
                self.burning_hands()
                self.wizard["spellslots"] = self.wizard["spellslots"] - 1
            # Attack (Cantrip)
            else:
                # Choose target for the attack
                target = self.choose_target()
                self.cantrip(target)

    def cleric_turn(self):
        # 0 Move
        # 1 Attack (Cantrip)
        # 2 Attack (Mace)
        # 3 Heal

        # Only Act if alive
        if self.wizard["alive"] and self.goblins_alive >= 1:
            # Choose target for the attack
            target = self.choose_target()

            # Move
            if self.cleric["hp"] < 5 and self.cleric["zone"] != 1:
                self.cleric["zone"] = 1
            # Heal
            elif self.fighter["hp"] < 5 and self.cleric["spellslots"] < 0:
                self.heal_ranged(self.fighter)
                self.cleric["spellslots"] = self.cleric["spellslots"] - 1
            elif self.rogue["hp"] < 5 and self.cleric["spellslots"] < 0:
                self.heal_ranged(self.rogue)
                self.cleric["spellslots"] = self.cleric["spellslots"] - 1
            elif self.wizard["hp"] < 5 and self.cleric["spellslots"] < 0:
                self.heal_ranged(self.wizard)
                self.cleric["spellslots"] = self.cleric["spellslots"] - 1
            elif self.cleric["hp"] > 5 and self.cleric["zone"] != 2:
                self.cleric["zone"] = 2
            # Attack (Cantrip)
            elif self.cleric["zone"] == 1:
                self.cantrip(target)
            # Attack (Mace)
            else:
                self.cleric_attack_melee(target)

    #####################################################################################

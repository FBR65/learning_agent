"""
Reinforcement Learning Agent für den Lernassistenten
"""

import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import os
from typing import Dict
import pickle

from .environment import LearningEnvironment
from .models import LearningScenario, LearnerState, AgentAction


class LearningProgressCallback(BaseCallback):
    """Callback um den Lernfortschritt des Agenten zu verfolgen"""

    def __init__(self, check_freq: int = 1000, save_path: str = "models/"):
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Speichere Modell-Checkpoint
            model_path = os.path.join(self.save_path, f"checkpoint_{self.n_calls}")
            self.model.save(model_path)

            # Logge Performance
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(
                    self.episode_rewards[-100:]
                )  # Letzten 100 Episoden
                print(f"Step {self.n_calls}: Mean Reward: {mean_reward:.2f}")

        return True

    def _on_rollout_end(self) -> None:
        # Sammle Episode-Statistiken - konvertiere sofort zu Python-Typen
        if hasattr(self.locals, "infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    # Stelle sicher, dass alle Werte Python-Typen sind
                    episode_reward = float(info["episode"]["r"])
                    episode_length = int(info["episode"]["l"])
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)


class LearningAgent:
    """
    Hauptklasse für den RL-basierten Lernagenten
    """

    def __init__(self, scenario: LearningScenario, algorithm: str = "DQN"):
        self.scenario = scenario
        self.algorithm = algorithm

        # Erstelle Umgebung
        self.env = LearningEnvironment(scenario)
        check_env(self.env)  # Validiere Gym-Umgebung

        # Erstelle RL-Modell
        self.model = None
        self.is_trained = False

        # Statistiken
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "best_reward": float("-inf"),
            "training_history": [],
        }

    def create_model(self, **kwargs):
        """Erstellt das RL-Modell basierend auf dem gewählten Algorithmus"""
        # Standard-Parameter für verschiedene Algorithmen
        default_params = {
            "DQN": {
                "learning_rate": 1e-3,
                "buffer_size": 10000,
                "learning_starts": 100,
                "batch_size": 32,
                "target_update_interval": 100,
                "exploration_fraction": 0.3,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "train_freq": 4,
                "gradient_steps": 1,
                "verbose": 1,
            },
            "A2C": {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
                "vf_coef": 0.25,
                "max_grad_norm": 0.5,
                "verbose": 1,
            },
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": 1,
            },
        }

        # Merge default params with user params
        params = default_params.get(self.algorithm, {})
        params.update(kwargs)

        # Erstelle Modell
        if self.algorithm == "DQN":
            self.model = DQN("MlpPolicy", self.env, **params)
        elif self.algorithm == "A2C":
            self.model = A2C("MlpPolicy", self.env, **params)
        elif self.algorithm == "PPO":
            self.model = PPO("MlpPolicy", self.env, **params)
        else:
            raise ValueError(f"Unbekannter Algorithmus: {self.algorithm}")

        print(f"Modell erstellt: {self.algorithm} mit Parametern: {params}")

    def train(self, total_timesteps: int = 10000, save_path: str = "models/"):
        """Trainiert den Agenten"""
        if self.model is None:
            self.create_model()

        # Setup Callback
        callback = LearningProgressCallback(
            check_freq=max(1, total_timesteps // 10), save_path=save_path
        )

        print(f"Starte Training für {total_timesteps} Timesteps...")

        # Training
        self.model.learn(
            total_timesteps=total_timesteps, callback=callback, progress_bar=True
        )

        self.is_trained = True

        # Update Statistiken - alle Werte zu Python-Typen konvertieren
        self.training_stats["total_steps"] += total_timesteps
        self.training_stats["episodes"] = len(callback.episode_rewards)
        if callback.episode_rewards:
            # Konvertiere numpy array zu Python float - sicher für alle Fälle
            rewards_list = [float(r) for r in callback.episode_rewards]
            self.training_stats["best_reward"] = float(max(rewards_list))

        # Speichere finales Modell
        final_path = os.path.join(
            save_path, f"final_{self.algorithm}_{self.scenario.name}"
        )
        self.model.save(final_path)

        print(f"Training abgeschlossen. Modell gespeichert unter: {final_path}")

        # Konvertiere ALLE Werte zu Python-Typen für JSON-Serialisierung
        episode_rewards_list = []
        episode_lengths_list = []

        for r in callback.episode_rewards:
            episode_rewards_list.append(float(r))

        for length in callback.episode_lengths:
            episode_lengths_list.append(int(length))

        return episode_rewards_list, episode_lengths_list

    def load_model(self, model_path: str):
        """Lädt ein vortrainiertes Modell"""
        if self.algorithm == "DQN":
            self.model = DQN.load(model_path, env=self.env)
        elif self.algorithm == "A2C":
            self.model = A2C.load(model_path, env=self.env)
        elif self.algorithm == "PPO":
            self.model = PPO.load(model_path, env=self.env)

        self.is_trained = True
        print(f"Modell geladen von: {model_path}")

    def get_action(self, learner_state: LearnerState) -> AgentAction:
        """
        Gibt die nächste Aktion des Agenten basierend auf dem Lernerzustand zurück
        """
        if not self.is_trained:
            raise ValueError(
                "Agent ist noch nicht trainiert! Rufe train() oder load_model() auf."
            )

        # Setze Umgebung in den aktuellen Zustand
        self.env.learner_state = learner_state
        obs = self.env._get_observation()

        # Vorhersage der Aktion
        action, _states = self.model.predict(obs, deterministic=True)

        # Konvertiere zu AgentAction
        agent_action = self.env._execute_action(int(action))

        return agent_action

    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """Evaluiert die Performance des Agenten"""
        if not self.is_trained:
            raise ValueError("Agent ist noch nicht trainiert!")

        episode_rewards = []
        episode_lengths = []
        success_rates = []

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0

            done = False
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

                if render:
                    print(
                        f"Schritt {episode_length}: Aktion={action}, Belohnung={reward:.2f}"
                    )

            # Berechne Erfolgsrate für diese Episode
            final_state = self.env.learner_state
            total_correct = sum(final_state.correct_answers.values())
            total_attempts = sum(
                final_state.get_total_attempts(cat) for cat in self.scenario.categories
            )
            success_rate = total_correct / max(total_attempts, 1)

            # Sammle Episode-Daten - GARANTIERT Python-Typen
            episode_reward = float(episode_reward)
            episode_length = int(episode_length)
            success_rate = float(success_rate)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success_rates.append(success_rate)

        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "mean_success_rate": float(np.mean(success_rates)),
            "episodes_evaluated": n_episodes,
        }

        print(f"Evaluation Ergebnisse ({n_episodes} Episoden):")
        print(
            f"  Durchschnittliche Belohnung: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
        )
        print(f"  Durchschnittliche Episode-Länge: {results['mean_length']:.1f}")
        print(f"  Durchschnittliche Erfolgsrate: {results['mean_success_rate']:.2%}")

        return results

    def save_stats(self, filepath: str):
        """Speichert Trainingsstatistiken"""
        with open(filepath, "wb") as f:
            pickle.dump(self.training_stats, f)

    def load_stats(self, filepath: str):
        """Lädt Trainingsstatistiken"""
        with open(filepath, "rb") as f:
            self.training_stats = pickle.load(f)

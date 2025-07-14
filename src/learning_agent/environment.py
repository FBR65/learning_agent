"""
Gymnasium-Umgebung für den Lernagenten
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime

from .models import LearningScenario, LearnerState, AgentAction, DifficultyLevel


class LearningEnvironment(gym.Env):
    """
    Gymnasium-Umgebung für den Lernagenten.

    Der Agent wählt Aktionen (welche Aufgabe stellen, Hinweise geben, etc.)
    basierend auf dem aktuellen Zustand des Lernenden.
    """

    def __init__(self, scenario: LearningScenario, max_session_length: int = 100):
        super().__init__()

        self.scenario = scenario
        self.max_session_length = max_session_length
        self.current_step = 0

        # Initialisiere Lernerzustand
        self.learner_state = None
        self.current_task = None
        self.session_history = []

        # Definiere Action Space
        # 0: Leichte Aufgabe stellen
        # 1: Mittlere Aufgabe stellen
        # 2: Schwere Aufgabe stellen
        # 3: Hinweis geben
        # 4: Feedback geben
        # 5: Schwierigkeit reduzieren
        self.action_space = spaces.Discrete(6)

        # Definiere Observation Space
        # State Vector: [success_rates_per_category, avg_response_times, hints_used_ratio,
        #                consecutive_correct, consecutive_incorrect, current_difficulty]
        self.num_categories = len(scenario.categories) if scenario.categories else 1
        obs_dim = (
            self.num_categories * 3 + 3
        )  # 3 features per category + 3 global features
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset der Umgebung für eine neue Lernsession"""
        super().reset(seed=seed)

        learner_id = (
            options.get("learner_id", "default_learner")
            if options
            else "default_learner"
        )

        self.learner_state = LearnerState(
            learner_id=learner_id, scenario_name=self.scenario.name
        )

        self.current_step = 0
        self.current_task = None
        self.session_history = []

        observation = self._get_observation()
        info = {"step": self.current_step, "learner_state": self.learner_state.dict()}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Führt eine Aktion aus und gibt den neuen Zustand, Belohnung und weitere Infos zurück
        """
        self.current_step += 1

        # Führe die gewählte Aktion aus
        agent_action = self._execute_action(action)

        # Simuliere Lerner-Response (in echter Anwendung würde dies vom UI kommen)
        learner_response = self._simulate_learner_response(agent_action)

        # Update Lerner-Zustand basierend auf Response
        self._update_learner_state(agent_action, learner_response)

        # Berechne Belohnung
        reward = self._calculate_reward(agent_action, learner_response)

        # Prüfe ob Episode beendet ist
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_session_length

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "action": agent_action.dict(),
            "learner_response": learner_response,
            "reward": reward,
            "learner_state": self.learner_state.dict(),
        }

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> AgentAction:
        """Konvertiert Aktions-ID in AgentAction"""
        action_types = {
            0: "present_easy_task",
            1: "present_medium_task",
            2: "present_hard_task",
            3: "give_hint",
            4: "provide_feedback",
            5: "reduce_difficulty",
        }

        # Konvertiere numpy array zu Python int
        action = int(action)
        action_type = action_types[action]
        parameters = {}

        if action_type.startswith("present_"):
            # Wähle passende Aufgabe basierend auf Schwierigkeit
            if action_type == "present_easy_task":
                difficulty = DifficultyLevel.BEGINNER
            elif action_type == "present_medium_task":
                difficulty = DifficultyLevel.INTERMEDIATE
            else:
                difficulty = DifficultyLevel.ADVANCED

            available_tasks = self.scenario.get_tasks_by_difficulty(difficulty)
            if available_tasks:
                selected_task = np.random.choice(available_tasks)
                self.current_task = selected_task
                parameters["task_id"] = selected_task.id
                parameters["question"] = selected_task.question
                parameters["difficulty"] = difficulty.value

        return AgentAction(
            action_type=action_type,
            parameters=parameters,
            timestamp=datetime.now().isoformat(),
        )

    def _simulate_learner_response(self, agent_action: AgentAction) -> Dict[str, Any]:
        """
        Simuliert eine Lerner-Antwort basierend auf aktueller Fähigkeit.
        In der echten Anwendung würde dies durch UI-Input ersetzt.
        """
        if not agent_action.action_type.startswith("present_"):
            return {"type": "acknowledgment"}

        if not self.current_task:
            return {"type": "no_response"}

        # Simuliere Antwort basierend auf geschätzter Fähigkeit
        category = self.current_task.category or "general"
        estimated_ability = self.learner_state.estimated_ability.get(category, 0.5)

        # Verwende Item Response Theory-ähnliche Wahrscheinlichkeit
        task_difficulty = (
            self.current_task.difficulty.value / 4.0
        )  # Normalisiere auf 0-1
        success_prob = 1 / (1 + np.exp(-(estimated_ability - task_difficulty)))

        is_correct = np.random.random() < success_prob
        response_time = np.random.normal(10, 5)  # Simuliere Antwortzeit

        return {
            "type": "task_response",
            "task_id": self.current_task.id,
            "is_correct": is_correct,
            "answer": self.current_task.answer if is_correct else "wrong_answer",
            "response_time": max(1, response_time),
            "hints_requested": np.random.random()
            < 0.3,  # 30% Chance auf Hinweis-Request
        }

    def _update_learner_state(
        self, agent_action: AgentAction, learner_response: Dict[str, Any]
    ):
        """Aktualisiert den Zustand des Lernenden basierend auf der Interaktion"""
        if learner_response["type"] != "task_response" or not self.current_task:
            return

        category = self.current_task.category or "general"

        # Update Antworten-Statistiken
        if learner_response["is_correct"]:
            self.learner_state.correct_answers[category] = (
                self.learner_state.correct_answers.get(category, 0) + 1
            )
            self.learner_state.consecutive_correct += 1
            self.learner_state.consecutive_incorrect = 0
        else:
            self.learner_state.incorrect_answers[category] = (
                self.learner_state.incorrect_answers.get(category, 0) + 1
            )
            self.learner_state.consecutive_incorrect += 1
            self.learner_state.consecutive_correct = 0

        # Update Antwortzeiten
        current_avg = self.learner_state.average_response_times.get(category, 0)
        total_attempts = self.learner_state.get_total_attempts(category)

        if total_attempts > 1:
            new_avg = (
                (current_avg * (total_attempts - 1)) + learner_response["response_time"]
            ) / total_attempts
        else:
            new_avg = learner_response["response_time"]

        self.learner_state.average_response_times[category] = new_avg

        # Update geschätzte Fähigkeit (vereinfachte IRT)
        current_ability = self.learner_state.estimated_ability.get(category, 0.5)
        task_difficulty = self.current_task.difficulty.value / 4.0

        learning_rate = 0.1
        if learner_response["is_correct"]:
            # Ability steigt, besonders wenn Aufgabe schwer war
            ability_change = learning_rate * (1 - current_ability) * task_difficulty
        else:
            # Ability sinkt, besonders wenn Aufgabe leicht war
            ability_change = -learning_rate * current_ability * (1 - task_difficulty)

        self.learner_state.estimated_ability[category] = np.clip(
            current_ability + ability_change, 0.0, 1.0
        )

    def _calculate_reward(
        self, agent_action: AgentAction, learner_response: Dict[str, Any]
    ) -> float:
        """Berechnet die Belohnung für die gewählte Aktion"""
        reward = 0.0

        if learner_response["type"] == "task_response":
            if learner_response["is_correct"]:
                # Positive Belohnung für korrekte Antworten
                reward += 1.0

                # Bonus für angemessene Schwierigkeit
                category = self.current_task.category or "general"
                success_rate = self.learner_state.get_success_rate(category)

                # Optimal ist eine Erfolgsrate von 70-80%
                if 0.7 <= success_rate <= 0.8:
                    reward += 0.5

                # Bonus für schnelle Antworten
                if learner_response["response_time"] < 5:
                    reward += 0.2

            else:
                # Negative Belohnung für falsche Antworten
                reward -= 0.5

                # Zusätzliche Strafe bei wiederholten Fehlern
                if self.learner_state.consecutive_incorrect > 2:
                    reward -= 0.3

        # Strafe für zu viele Hinweise
        if agent_action.action_type == "give_hint":
            reward -= 0.1

        # Belohnung für angemessene Schwierigkeitsanpassung
        if (
            agent_action.action_type == "reduce_difficulty"
            and self.learner_state.consecutive_incorrect > 1
        ):
            reward += 0.3

        return reward

    def _is_terminated(self) -> bool:
        """Prüft ob die Episode beendet werden soll"""
        # Beende wenn Lerner sehr frustriert ist (viele falsche Antworten hintereinander)
        if self.learner_state.consecutive_incorrect > 5:
            return True

        # Beende wenn sehr gute Performance erreicht wurde
        total_attempts = sum(
            self.learner_state.get_total_attempts(cat)
            for cat in self.scenario.categories
        )
        if total_attempts > 20:
            avg_success = np.mean(
                [
                    self.learner_state.get_success_rate(cat)
                    for cat in self.scenario.categories
                ]
            )
            if avg_success > 0.9:
                return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Konvertiert den aktuellen Zustand in einen Observations-Vektor"""
        obs = []

        categories = (
            self.scenario.categories if self.scenario.categories else ["general"]
        )

        for category in categories:
            # Success rate für diese Kategorie
            success_rate = self.learner_state.get_success_rate(category)
            obs.append(success_rate)

            # Durchschnittliche Antwortzeit (normalisiert)
            avg_time = self.learner_state.average_response_times.get(category, 10.0)
            obs.append(min(avg_time / 30.0, 1.0))  # Normalisiere auf max 30 Sekunden

            # Hints-zu-Versuche Ratio
            hints = self.learner_state.hints_used.get(category, 0)
            attempts = self.learner_state.get_total_attempts(category)
            hints_ratio = hints / max(attempts, 1)
            obs.append(hints_ratio)

        # Globale Features
        obs.append(min(self.learner_state.consecutive_correct / 10.0, 1.0))
        obs.append(min(self.learner_state.consecutive_incorrect / 5.0, 1.0))

        # Aktuelle geschätzte Fähigkeit (Durchschnitt über alle Kategorien)
        abilities = list(self.learner_state.estimated_ability.values())
        avg_ability = np.mean(abilities) if abilities else 0.5
        obs.append(avg_ability)

        return np.array(obs, dtype=np.float32)
